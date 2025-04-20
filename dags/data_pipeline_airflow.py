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
def _run_ms_pipeline(**kwargs):
    # Import happens when the task runs (on the worker)
    from data_pipeline.ms import run_ms_pipeline
    run_ms_pipeline(**kwargs) # Pass along any context/kwargs if needed

def _run_blog_pipeline(**kwargs):
    from data_pipeline.blogs import run_blog_pipeline
    run_blog_pipeline()

def _run_pdf_pipeline(**kwargs):
    from data_pipeline.pdfs import run_pdf_pipeline
    run_pdf_pipeline() # Add op_kwargs in PythonOperator if limit is needed

def _run_other_preprocess_pipeline(**kwargs):
    from data_pipeline.other_preprocesing import run_other_preprocess_pipeline
    run_other_preprocess_pipeline()

def _run_ms_preprocess_pipeline(**kwargs):
    from data_pipeline.ms_preprocess import run_ms_preprocess_pipeline
    run_ms_preprocess_pipeline()

def _run_chunk_embed_store_pipeline(**kwargs):
    # This function is called by the worker, which *has* langchain installed
    from data_pipeline.vector_db import run_chunk_embed_store_pipeline
    run_chunk_embed_store_pipeline()

# --- DAG Definition ---
default_args = {
    'owner': 'vinyas',
    'start_date': pendulum.datetime(2024, 4, 18, tz="UTC"),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
        dag_id="Data_pipeline_dag",
        default_args=default_args,
        description='Scrapes data, preprocesses, stores artifacts in GCS, stores embeddings in Pinecone, triggers evaluation.',
        schedule='@weekly',
        catchup=False,
        tags=["data_pipeline", "etl", "rag", "gcs", "pinecone"],
) as dag:

    # --- Define REAL Tasks using the wrapper functions ---
    scrape_ms_task = PythonOperator(
        task_id='scrape_ms_task',
        python_callable=_run_ms_pipeline, # Call the wrapper
    )

    scrape_blog_task = PythonOperator(
        task_id='scrape_blog_task',
        python_callable=_run_blog_pipeline,
    )

    scrape_pdf_task = PythonOperator(
        task_id='scrape_pdf_task',
        python_callable=_run_pdf_pipeline,
    )

    preprocess_ms_task = PythonOperator(
        task_id='preprocess_ms_task',
        python_callable=_run_ms_preprocess_pipeline,
    )

    preprocess_other_data_task = PythonOperator(
        task_id='preprocess_other_data_task',
        python_callable=_run_other_preprocess_pipeline,
    )

    chunk_embed_store_task = PythonOperator(
        task_id='chunk_embed_store_pinecone_task',
        python_callable=_run_chunk_embed_store_pipeline, # Call wrapper
        execution_timeout=timedelta(hours=2),
    )

    trigger_evaluation_dag = TriggerDagRunOperator(
        task_id="trigger_model_evaluation_pipeline",
        trigger_dag_id="model_evaluation_pipeline_dag",
        conf={"triggering_run_id": "{{ run_id }}"},
        execution_date="{{ dag_run.logical_date }}",
        reset_dag_run=True,
        wait_for_completion=False,
    )

    # --- Define Task Dependencies ---
    scrape_ms_task >> preprocess_ms_task
    [scrape_blog_task, scrape_pdf_task] >> preprocess_other_data_task
    [preprocess_ms_task, preprocess_other_data_task] >> chunk_embed_store_task
    chunk_embed_store_task >> trigger_evaluation_dag