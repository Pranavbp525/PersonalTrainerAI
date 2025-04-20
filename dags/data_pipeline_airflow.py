# dags/data_pipeline_airflow.py
import pendulum
import sys
import os
import logging
from datetime import datetime, timedelta

# Airflow specific imports
from airflow.models.dag import DAG # Use DAG from models for Airflow 2+
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator # Keep import just in case needed elsewhere
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

# --- Add src directory to Python path ---
# Adjust path based on the new docker-compose mount point
SRC_PATH = "/opt/airflow/app/src" # Path inside container where src is mounted
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)
    logging.info(f"Appended {SRC_PATH} to sys.path")
else:
    logging.info(f"{SRC_PATH} already in sys.path")


# --- Import refactored pipeline functions ---
# It's good practice to handle potential import errors
try:
    from data_pipeline.ms import run_ms_pipeline
    from data_pipeline.blogs import run_blog_pipeline
    from data_pipeline.pdfs import run_pdf_pipeline
    from data_pipeline.other_preprocesing import run_other_preprocess_pipeline
    from data_pipeline.ms_preprocess import run_ms_preprocess_pipeline
    # Import the correct function name from vector_db.py
    from data_pipeline.vector_db import run_chunk_embed_store_pipeline
    IMPORT_SUCCESS = True
    logging.info("Successfully imported pipeline functions.")
except ImportError as e:
    logging.error(f"Failed to import pipeline functions: {e}", exc_info=True)
    IMPORT_SUCCESS = False
    # Define dummy functions if import fails to allow DAG parsing
    def run_ms_pipeline(**kwargs): raise ImportError("ms module not found")
    def run_blog_pipeline(**kwargs): raise ImportError("blogs module not found")
    def run_pdf_pipeline(**kwargs): raise ImportError("pdfs module not found")
    def run_other_preprocess_pipeline(**kwargs): raise ImportError("other_preprocessing module not found")
    def run_ms_preprocess_pipeline(**kwargs): raise ImportError("ms_preprocess module not found")
    def run_chunk_embed_store_pipeline(**kwargs): raise ImportError("vector_db module not found")


# --- DAG Definition ---

# Define DVC remote name used in configuration (still needed if DVC is used elsewhere)
# DVC_REMOTE_NAME = "gcs-remote"

default_args = {
    'owner': 'vinyas', # Changed owner
    'start_date': pendulum.datetime(2024, 4, 18, tz="UTC"), # Use pendulum
    'retries': 1, # Default retries
    'retry_delay': timedelta(minutes=5), # Default retry delay
    # Email config commented out, enable if your Airflow SMTP is configured
    # 'email_on_failure': True,
    # 'email': ['your_email@example.com'],
    # 'email_on_retry': False,
}

with DAG(
        dag_id="Data_pipeline_dag", # Ensure this ID is correct and unique
        default_args=default_args,
        description='Scrapes data, preprocesses, stores artifacts in GCS, stores embeddings in Pinecone, triggers evaluation.', # Removed DVC from desc
        schedule='@weekly', # Keep your schedule
        catchup=False,
        tags=["data_pipeline", "etl", "rag", "gcs", "pinecone"], # Removed DVC tag
) as dag:

    if not IMPORT_SUCCESS:
        # If imports failed, create a single dummy task to indicate the error
        import_error_task = BashOperator(
            task_id='import_error',
            bash_command='echo "CRITICAL: Failed to import pipeline functions. Check Airflow logs and sys.path configuration." && exit 1',
        )
    else:
        # --- Scraping Tasks ---
        scrape_ms_task = PythonOperator(
            task_id='scrape_ms_task',
            python_callable=run_ms_pipeline,
        )

        scrape_blog_task = PythonOperator(
            task_id='scrape_blog_task',
            python_callable=run_blog_pipeline,
        )

        scrape_pdf_task = PythonOperator(
            task_id='scrape_pdf_task',
            python_callable=run_pdf_pipeline,
        )

        # --- Preprocessing Tasks ---
        # These read from GCS raw bucket and write to GCS processed bucket
        preprocess_ms_task = PythonOperator(
            task_id='preprocess_ms_task',
            python_callable=run_ms_preprocess_pipeline,
        )

        preprocess_other_data_task = PythonOperator(
            task_id='preprocess_other_data_task',
            python_callable=run_other_preprocess_pipeline,
        )

        # --- DVC Pull Task - REMOVED ---
        # dvc_pull_processed_task = BashOperator(...) # Definition removed

        # --- Chunking, Embedding, Storing Task ---
        # Reads preprocessed data directly from GCS processed bucket
        # Needs PINECONE_API_KEY from environment
        chunk_embed_store_task = PythonOperator(
            task_id='chunk_embed_store_pinecone_task', # Kept original task_id for consistency
            python_callable=run_chunk_embed_store_pipeline, # Use correct function name
            execution_timeout=timedelta(hours=2), # Adjust timeout as needed for embedding
            doc_md="Chunks preprocessed data from GCS, generates embeddings, and stores in Pinecone.",
        )

        # --- Trigger Next DAG Task ---
        # Check if trigger_dag_id ('model_evaluation_pipeline_dag') matches the actual dag_id in the other file
        trigger_evaluation_dag = TriggerDagRunOperator(
            task_id="trigger_model_evaluation_pipeline",
            trigger_dag_id="model_evaluation_pipeline_dag", # <<< VERIFY THIS DAG_ID
            conf={"triggering_run_id": "{{ run_id }}"}, # Pass context if needed
            execution_date="{{ dag_run.logical_date }}",
            reset_dag_run=True,
            wait_for_completion=False,
            doc_md="Triggers the model evaluation DAG.",
        )

        # --- Define Task Dependencies ---
        # Scraping tasks can run in parallel
        # Preprocessing tasks depend on their respective scrape task
        scrape_ms_task >> preprocess_ms_task
        # Other preprocessing depends on blogs and PDFs scrape finishing
        [scrape_blog_task, scrape_pdf_task] >> preprocess_other_data_task

        # --- UPDATED DEPENDENCY ---
        # Chunking/Embedding/Storing depends DIRECTLY on preprocessing finishing
        [preprocess_ms_task, preprocess_other_data_task] >> chunk_embed_store_task

        # Triggering the next DAG depends on successful chunking/embedding/storing
        chunk_embed_store_task >> trigger_evaluation_dag