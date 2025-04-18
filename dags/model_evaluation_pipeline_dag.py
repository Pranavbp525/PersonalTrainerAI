# dags/model_evaluation_pipeline_dag.py

from __future__ import annotations
import pendulum
import os
import logging
import traceback
from datetime import timedelta

# Import necessary Airflow classes FIRST
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
# --- REMOVED Sensor and State imports ---
# from airflow.sensors.external_task import ExternalTaskSensor
# from airflow.utils.state import DagRunState

# Import dotenv AFTER airflow imports
from dotenv import load_dotenv

# --- Configuration ---
PROJECT_ROOT = "/opt/airflow"
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
EVAL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "airflow_eval_output")

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

import sys
if SRC_DIR not in sys.path:
    log.info(f"Adding {SRC_DIR} to sys.path for DAG definition/worker.")
    sys.path.insert(0, SRC_DIR)

# --- Task Function for RAG Evaluation ---
def run_rag_evaluation_task(**context):
    # ... (Keep the inside of this function exactly as it was) ...
    # Including the load_dotenv() call at the beginning
    task_log = logging.getLogger("airflow.task")
    task_log.info("--- Starting RAG Evaluation Task ---")
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(env_path):
        loaded = load_dotenv(dotenv_path=env_path, override=True, verbose=True)
        task_log.info(f".env file found at {env_path}, load_dotenv result: {loaded}")
        task_log.info(f"MLFLOW_TRACKING_URI from os.getenv: {os.getenv('MLFLOW_TRACKING_URI')}")
        task_log.info(f"OPENAI_API_KEY from os.getenv: {os.getenv('OPENAI_API_KEY')}")
        task_log.info(f"PINECONE_API_KEY from os.getenv: {os.getenv('PINECONE_API_KEY')}")
    else:
        task_log.warning(f".env file not found at {env_path}. Relying on pre-set environment variables.")
    # ... (rest of the function is unchanged) ...
    required_vars = {
        "MLFLOW_TRACKING_URI": os.getenv('MLFLOW_TRACKING_URI'),
        "OPENAI_API_KEY": os.getenv('OPENAI_API_KEY')
        # "PINECONE_API_KEY": os.getenv('PINECONE_API_KEY') # Add back if needed by RAGEvaluator directly
    }
    missing_vars = [k for k, v in required_vars.items() if not v]
    if missing_vars:
        error_msg = f"RAG Eval Task Failed: Missing required environment variables after attempting load: {', '.join(missing_vars)}."
        task_log.error(error_msg)
        raise ValueError(error_msg)
    else:
        task_log.info("RAG Eval Task: Required environment variables are present.")
        task_log.info(f"MLFLOW_TRACKING_URI detected: {required_vars['MLFLOW_TRACKING_URI']}")
    try:
        os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
        task_log.info(f"Ensured output directory exists: {EVAL_OUTPUT_DIR}")
    except OSError as e:
        task_log.error(f"Could not create output directory {EVAL_OUTPUT_DIR}: {e}")
        raise
    try:
        from src.rag_model.advanced_rag_evaluation import AdvancedRAGEvaluator
        task_log.info("Initializing AdvancedRAGEvaluator...")
        evaluator = AdvancedRAGEvaluator(output_dir=EVAL_OUTPUT_DIR)
        task_log.info("Running comparison of RAG implementations via evaluator.compare_implementations()...")
        comparison_results = evaluator.compare_implementations()
        task_log.info("RAG Implementation Comparison Finished.")
        if comparison_results and comparison_results.get("best_implementation"):
            task_log.info(f"Best RAG Implementation determined: {comparison_results['best_implementation']}")
            task_log.info(f"Best Overall Score: {comparison_results.get('best_score', 'N/A'):.2f}")
            evaluator.print_summary(comparison_results)
        else:
            task_log.warning("Could not determine best RAG implementation from results, or results were empty.")
        task_log.info("--- RAG Evaluation Task Successfully Completed ---")
        return True
    except ImportError as e:
        task_log.error(f"Import Error during RAG evaluation task: {e}")
        task_log.error(f"Current sys.path: {sys.path}")
        raise
    except Exception as e:
        task_log.error(f"An error occurred during RAG evaluation task: {e}")
        task_log.error(traceback.format_exc())
        raise

# --- Task Function for Agent Evaluation ---
def run_agent_evaluation_task(**context):
     # ... (Keep the inside of this function exactly as it was) ...
     # Including the load_dotenv() call at the beginning
    task_log = logging.getLogger("airflow.task")
    task_log.info("--- Starting Agent Evaluation Task ---")
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(env_path):
        loaded = load_dotenv(dotenv_path=env_path, override=True, verbose=True)
        task_log.info(f".env file found at {env_path}, load_dotenv result: {loaded}")
        task_log.info(f"MLFLOW_TRACKING_URI from os.getenv: {os.getenv('MLFLOW_TRACKING_URI')}")
        task_log.info(f"LANGSMITH_API_KEY from os.getenv: {os.getenv('LANGSMITH_API_KEY')}")
        task_log.info(f"OPENAI_API_KEY from os.getenv: {os.getenv('OPENAI_API_KEY')}")
    else:
        task_log.warning(f".env file not found at {env_path}. Relying on pre-set environment variables.")
    # ... (rest of the function is unchanged) ...
    required_vars = {
        "MLFLOW_TRACKING_URI": os.getenv('MLFLOW_TRACKING_URI'),
        "LANGSMITH_API_KEY": os.getenv('LANGSMITH_API_KEY'),
        "LANGSMITH_PROJECT": os.getenv('LANGSMITH_PROJECT'),
        "OPENAI_API_KEY": os.getenv('OPENAI_API_KEY')
    }
    missing_vars = [k for k, v in required_vars.items() if not v]
    if missing_vars:
        error_msg = f"Agent Eval Task Failed: Missing required environment variables after attempting load: {', '.join(missing_vars)}."
        task_log.error(error_msg)
        raise ValueError(error_msg)
    else:
        task_log.info("Agent Eval Task: Required environment variables are present.")
        task_log.info(f"MLFLOW_TRACKING_URI detected: {required_vars['MLFLOW_TRACKING_URI']}")
        task_log.info(f"LANGSMITH_PROJECT detected: {required_vars['LANGSMITH_PROJECT']}")
    try:
        from src.chatbot.agent_eval.eval import evaluate_agent
        task_log.info("Running Agent evaluation via evaluate_agent()...")
        average_accuracy = evaluate_agent()
        if average_accuracy is not None:
            task_log.info(f"Agent evaluation completed. Average Accuracy reported: {average_accuracy:.2f}")
        else:
            task_log.warning("Agent evaluation finished, but no average accuracy score was returned/calculated.")
        task_log.info("--- Agent Evaluation Task Successfully Completed ---")
        return True
    except ImportError as e:
        task_log.error(f"Import Error during Agent evaluation task: {e}")
        task_log.error(f"Current sys.path: {sys.path}")
        raise
    except Exception as e:
        task_log.error(f"An error occurred during Agent evaluation task: {e}")
        task_log.error(traceback.format_exc())
        raise

# --- Define DAG ---
with DAG(
    dag_id='model_evaluation_pipeline',
    default_args={
        'owner': 'Your Name / Team', # CHANGE THIS
        'start_date': pendulum.datetime(2024, 4, 17, tz="UTC"),
        'depends_on_past': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=3),
        'catchup': False,
        'email_on_failure': False,
        'email_on_retry': False,
    },
    description='Runs RAG and Agent evaluations using MLflow. Triggered by Data_pipeline_dag.',
    schedule=None, # This DAG is triggered, not scheduled
    tags=['evaluation', 'rag', 'agent', 'mlflow'],
    template_searchpath=PROJECT_ROOT
) as dag:

    # --- REMOVED ExternalTaskSensor ---
    # wait_for_data_pipeline = ExternalTaskSensor(...)

    # Task 1 (was Task 2): Run RAG Evaluation Comparison
    rag_evaluation_task = PythonOperator(
        task_id='run_rag_evaluation',
        python_callable=run_rag_evaluation_task,
    )

    # Task 2 (was Task 3): Run Agent Evaluation
    agent_evaluation_task = PythonOperator(
        task_id='run_agent_evaluation',
        python_callable=run_agent_evaluation_task,
    )

    # Define task dependencies: Run evaluations in parallel AFTER DAG start
    # (No upstream dependency within this DAG anymore)
    # If you wanted them sequential: rag_evaluation_task >> agent_evaluation_task
    # For parallel: just define them without internal dependencies