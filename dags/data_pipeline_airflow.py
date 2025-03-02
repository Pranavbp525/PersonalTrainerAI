import sys
import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
# # Append the absolute path to src directory
# sys.path.append(src_path)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from data_pipeline.ms_preprocess import ms_preprocessing
from data_pipeline.ms import ms_scraper
from data_pipeline.blogs import blog_scraper
from data_pipeline.pdfs import pdf_scraper
from data_pipeline.other_preprocesing import preprocess_json_other_files
from data_pipeline.vector_db import chunk_to_db


default_args = {
    'owner': 'ruthvika',
    'start_date': datetime(2025, 3, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
    'email': ['ruthvikareddytangirala20@gmail.com'],
    'email_on_retry': False,
}

with DAG(
        'Data_pipeline_dag',
        default_args=default_args,
        description='A data pipeline DAG that scrapes sections, preprocesses the data',
        #schedule_interval='@daily',
        catchup=False
) as dag:
    def scrape_ms_website():
        ms_scraper()


    scrape_ms_task = PythonOperator(
        task_id='scrape_ms_task',
        python_callable=scrape_ms_website,
    )

    def preprocess_ms_data():

        ms_preprocessing()


    preprocess_ms_task = PythonOperator(
        task_id='preprocess_ms_task',
        python_callable=preprocess_ms_data,
    )

    def scrape_blogs():
        blog_scraper()


    scrape_blog_task = PythonOperator(
        task_id='scrape_blog_task',
        python_callable=scrape_blogs,
    )

    def scrape_pdfs():
        pdf_scraper()


    scrape_pdf_task = PythonOperator(
        task_id='scrape_pdf_task',
        python_callable=scrape_pdfs,
    )

    def preprocess_other_data():
        preprocess_json_other_files()


    preprocess_other_data_task = PythonOperator(
        task_id='preprocess_other_data_task',
        python_callable=preprocess_other_data,
    )

    def vectordb_pinecone():
        chunk_to_db()


    chunk_db_task = PythonOperator(
        task_id='chunk_db_task',
        python_callable=vectordb_pinecone,
    )



    # def upload_task_func():
    #     # Iterate through all processed files in the directory
    #     for root, _, files in os.walk(PROCESSED_DATA_PATH):
    #         for file in files:
    #             if file.endswith(".json"):  # Assuming you want to upload .txt files
    #                 file_path = os.path.join(root, file)
    #                 upload_to_blob(file_path, file)  # Upload each processed file
    #
    #
    # blob_storage_task = PythonOperator(
    #     task_id='blob_storage_task',
    #     python_callable=upload_task_func,
    # )
    #
    # index_task = PythonOperator(
    #     task_id='index_task',
    #     python_callable=index_data_in_search,
    # )

    # ✅ Define task dependencies for parallel execution
    scrape_ms_task >> preprocess_ms_task  # (1) MS data processing

    [scrape_blog_task, scrape_pdf_task] >> preprocess_other_data_task  # (2) & (3) run in parallel and then merge

    [preprocess_ms_task, preprocess_other_data_task] >> chunk_db_task  # ✅ Final merge before storing in DB
