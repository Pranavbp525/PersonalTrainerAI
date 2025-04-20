# dags/data_pipeline_airflow.py
import pendulum
import sys
import os
import logging
from datetime import datetime, timedelta

from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

# --- Add src directory to Python path ---
SRC_PATH = "/opt/airflow/app/src" # Path inside container where src is mounted
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)
    logging.info(f"Appended {SRC_PATH} to sys.path")

# --- Define Wrapper Functions (Imports happen inside) ---
# --- UPDATED: Wrappers no longer pass **kwargs down ---
def _run_ms_pipeline(**context): # Still accepts context from Airflow
    # Import happens when the task runs (on the worker)
    from data_pipeline.ms import run_ms_pipeline
    # Call the function WITHOUT passing Airflow context (**kwargs or **context)
    # Assumes run_ms_pipeline uses its internal defaults or needs no specific args from Airflow context
    run_ms_pipeline()

def _run_blog_pipeline(**context): # Still accepts context from Airflow
    from data_pipeline.blogs import run_blog_pipeline
    # Call the function WITHOUT passing Airflow context
    run_blog_pipeline()

def _run_pdf_pipeline(**context): # Still accepts context from Airflow
    from data_pipeline.pdfs import run_pdf_pipeline
    # Call the function WITHOUT passing Airflow context
    # If run_pdf_pipeline needs 'limit', it must be passed via op_kwargs
    # in the PythonOperator definition below, e.g., op_kwargs={'limit': 5}
    # and then accessed here via context['params']['limit'] if needed.
    # For now, calling without args, assuming internal default is None (no limit).
    run_pdf_pipeline()

def _run_other_preprocess_pipeline(**context): # Still accepts context from Airflow
    from data_pipeline.other_preprocesing import run_other_preprocess_pipeline
    # Call the function WITHOUT passing Airflow context
    run_other_preprocess_pipeline()

def _run_ms_preprocess_pipeline(**context): # Still accepts context from Airflow
    from data_pipeline.ms_preprocess import run_ms_preprocess_pipeline
    # Call the function WITHOUT passing Airflow context
    run_ms_preprocess_pipeline()

def _run_chunk_embed_store_pipeline(**context): # Still accepts context from Airflow
    # This function is called by the worker, which *has* langchain installed
    from data_pipeline.vector_db import run_chunk_embed_store_pipeline
    # Call the function WITHOUT passing Airflow context
    run_chunk_embed_store_pipeline()

# --- DAG Definition ---
default_args = {
    'owner': 'vinyas',
    'start_date': pendulum.datetime(2024, 4, 18, tz="UTC"),
    'retries': 1, # Default to 1 retry as per original code
    'retry_delay': timedelta(minutes=5),
}

with DAG(
        dag_id="Data_pipeline_dag",
        default_args=default_args,
        description='Scrapes data, preprocesses, stores artifacts in GCS, stores embeddings in Pinecone, triggers evaluation.',
        schedule='@weekly', # Or None if manually triggered
        catchup=False,
        tags=["data_pipeline", "etl", "rag", "gcs", "pinecone"],
) as dag:

    # --- Define REAL Tasks using the wrapper functions ---
    scrape_ms_task = PythonOperator(
        task_id='scrape_ms_task',
        python_callable=_run_ms_pipeline, # Call the updated wrapper
        # If you need to override the default max_workouts in ms.py,
        # pass it here and modify _run_ms_pipeline to use context['params']
        # op_kwargs={'max_workouts': 10},
    )

    scrape_blog_task = PythonOperator(
        task_id='scrape_blog_task',
        python_callable=_run_blog_pipeline, # Call the updated wrapper
    )

    scrape_pdf_task = PythonOperator(
        task_id='scrape_pdf_task',
        python_callable=_run_pdf_pipeline, # Call the updated wrapper
        # If you need to override the default limit in pdfs.py,
        # pass it here and modify _run_pdf_pipeline to use context['params']
        # op_kwargs={'limit': 5},
    )

    preprocess_ms_task = PythonOperator(
        task_id='preprocess_ms_task',
        python_callable=_run_ms_preprocess_pipeline, # Call the updated wrapper
    )

    preprocess_other_data_task = PythonOperator(
        task_id='preprocess_other_data_task',
        python_callable=_run_other_preprocess_pipeline, # Call the updated wrapper
    )

    chunk_embed_store_task = PythonOperator(
        task_id='chunk_embed_store_pinecone_task',
        python_callable=_run_chunk_embed_store_pipeline, # Call the updated wrapper
        execution_timeout=timedelta(hours=2), # Keep execution timeout
    )

    trigger_evaluation_dag = TriggerDagRunOperator(
        task_id="trigger_model_evaluation_pipeline",
        trigger_dag_id="model_evaluation_pipeline_dag", # Verify this ID matches the eval DAG
        conf={"triggering_run_id": "{{ run_id }}"},
        execution_date="{{ dag_run.logical_date }}",
        reset_dag_run=True,
        wait_for_completion=False,
    )

    # --- Define Task Dependencies ---
    # These remain the same
    scrape_ms_task >> preprocess_ms_task
    [scrape_blog_task, scrape_pdf_task] >> preprocess_other_data_task
    [preprocess_ms_task, preprocess_other_data_task] >> chunk_embed_store_task
    chunk_embed_store_task >> trigger_evaluation_dag