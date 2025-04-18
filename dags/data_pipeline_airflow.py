# dags/data_pipeline_airflow.py

import sys
import os
from airflow import DAG
from airflow.operators.python import PythonOperator
import logging
# --- ADD THIS IMPORT ---
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
# --- END ADD IMPORT ---
from datetime import datetime, timedelta
from dotenv import load_dotenv # Keep this if tasks use it

# Add src directory (keep your existing logic)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from data_pipeline.ms_preprocess import ms_preprocessing
from data_pipeline.ms import ms_scraper
from data_pipeline.blogs import blog_scraper
from data_pipeline.pdfs import pdf_scraper
from data_pipeline.other_preprocesing import preprocess_json_other_files
from data_pipeline.vector_db import chunk_to_db

# Define PROJECT_ROOT inside the DAG file scope if needed for env path
PROJECT_ROOT_FOR_ENV = "/opt/airflow" # Path inside container

default_args = {
    'owner': 'ruthvika', # Consider changing owner if needed
    'start_date': datetime(2025, 3, 1), # Adjust start date if needed
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
    'email': ['ruthvikareddytangirala20@gmail.com'],
    'email_on_retry': False,
}

with DAG(
        'Data_pipeline_dag', # Ensure this ID is correct
        default_args=default_args,
        description='A data pipeline DAG that scrapes sections, preprocesses the data, and triggers evaluation.',
        schedule=None, # Keep as None if you trigger this manually
        # schedule_interval='@daily', # Or set a schedule if desired
        catchup=False
) as dag:

    # --- Define Task Functions (Load .env at start if needed) ---
    def load_env_if_needed(task_log):
        """Helper to load .env at task start."""
        env_path = os.path.join(PROJECT_ROOT_FOR_ENV, ".env")
        if os.path.exists(env_path):
            loaded = load_dotenv(dotenv_path=env_path, override=True, verbose=True)
            task_log.info(f".env file found at {env_path}, load_dotenv result: {loaded}")
            return True
        else:
            task_log.warning(f".env file not found at {env_path}. Relying on pre-set environment variables.")
            return False

    def scrape_ms_website():
        task_log = logging.getLogger("airflow.task")
        load_env_if_needed(task_log) # Load .env if ms_scraper needs it
        ms_scraper()

    scrape_ms_task = PythonOperator(
        task_id='scrape_ms_task',
        python_callable=scrape_ms_website,
    )

    def preprocess_ms_data():
        task_log = logging.getLogger("airflow.task")
        load_env_if_needed(task_log) # Load .env if ms_preprocessing needs it
        ms_preprocessing()

    preprocess_ms_task = PythonOperator(
        task_id='preprocess_ms_task',
        python_callable=preprocess_ms_data,
    )

    def scrape_blogs():
        task_log = logging.getLogger("airflow.task")
        load_env_if_needed(task_log) # Load .env if blog_scraper needs it
        blog_scraper()

    scrape_blog_task = PythonOperator(
        task_id='scrape_blog_task',
        python_callable=scrape_blogs,
    )

    def scrape_pdfs():
        task_log = logging.getLogger("airflow.task")
        load_env_if_needed(task_log) # Load .env if pdf_scraper needs it
        pdf_scraper()

    scrape_pdf_task = PythonOperator(
        task_id='scrape_pdf_task',
        python_callable=scrape_pdfs,
    )

    def preprocess_other_data():
        task_log = logging.getLogger("airflow.task")
        load_env_if_needed(task_log) # Load .env if preprocess_json_other_files needs it
        preprocess_json_other_files()

    preprocess_other_data_task = PythonOperator(
        task_id='preprocess_other_data_task',
        python_callable=preprocess_other_data,
    )

    def vectordb_pinecone():
        task_log = logging.getLogger("airflow.task")
        load_env_if_needed(task_log) # Load .env before calling chunk_to_db
        # Optional: Confirm key loaded
        task_log.info(f"PINECONE_API_KEY from os.getenv in DAG task: {os.getenv('PINECONE_API_KEY')}")
        chunk_to_db() # This function now uses os.getenv internally

    chunk_db_task = PythonOperator(
        task_id='chunk_db_task',
        python_callable=vectordb_pinecone,
    )

    # --- ADD TRIGGER DAG TASK ---
    trigger_evaluation_dag = TriggerDagRunOperator(
        task_id="trigger_model_evaluation_pipeline",
        trigger_dag_id="model_evaluation_pipeline",  # DAG ID of the DAG to trigger
        # conf={"triggering_run_id": "{{ run_id }}"}, # Optional: Pass context like the run_id
        execution_date="{{ dag_run.logical_date }}", # Trigger for the same logical date
        reset_dag_run=True, # If a run already exists for this date, reset and run again
        wait_for_completion=False, # Don't wait for the evaluation DAG to finish
    )
    # --- END TRIGGER DAG TASK ---


    # Define task dependencies
    scrape_ms_task >> preprocess_ms_task
    [scrape_blog_task, scrape_pdf_task] >> preprocess_other_data_task
    # The last task (chunk_db_task) should trigger the next DAG
    [preprocess_ms_task, preprocess_other_data_task] >> chunk_db_task >> trigger_evaluation_dag