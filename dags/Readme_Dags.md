# Airflow DAGs for PersonalTrainerAI

This directory contains the Apache Airflow DAG (Directed Acyclic Graph) definitions used to orchestrate the data processing and model evaluation pipelines for the PersonalTrainerAI project.

## Overview

Airflow is used here to automate, schedule, and monitor the project's workflows. This ensures that data scraping, preprocessing, vector database loading, and subsequent model/agent evaluations run in the correct order and can be easily monitored.

The primary workflows orchestrated are:

1.  **Data Pipeline:** Scrapes data from various sources (websites, blogs, PDFs), preprocesses it, and loads it into a vector database (e.g., Pinecone).
2.  **Model Evaluation Pipeline:** Evaluates the performance of different RAG implementations and the conversational agent, logging results to MLflow. This pipeline automatically runs after the data pipeline completes successfully.

## Prerequisites

Before running the Airflow stack, ensure you have the following installed on your host machine:

*   **Docker:** [Install Docker](https://docs.docker.com/get-docker/)
*   **Docker Compose:** Usually included with Docker Desktop. Verify with `docker-compose --version`.
*   **Git:** To clone the project repository.
*   **Project Code:** The `PersonalTrainerAI` project cloned locally.
*   **`.env` File:** A properly configured `.env` file located at the **root** of the `PersonalTrainerAI` project directory (see [Configuration](#configuration-env-file) section below).

## Airflow Setup (Docker Compose)

The Airflow environment is defined and managed using the `docker-compose.yaml` file located at the project root.

### Key Components:

*   **Services:** It launches the necessary Airflow components (webserver, scheduler, worker, triggerer) using the CeleryExecutor, along with dependent services (PostgreSQL metadata database, Redis Celery broker).
*   **Image:** It uses a specific official `apache/airflow` image tag (e.g., `apache/airflow:2.9.2-python3.9`). Using an image with Python 3.9+ is **crucial** for compatibility with libraries like Langchain, OpenAI, etc. This is configured in the `docker-compose.yaml` file under `x-airflow-common.image`.
*   **Volumes:** The following project directories are mounted into the Airflow containers:
    *   `./dags:/opt/airflow/dags` (Makes these DAG files visible to Airflow)
    *   `./src:/opt/airflow/src` (Allows DAG tasks to import custom Python code)
    *   `./logs:/opt/airflow/logs` (Stores Airflow task logs)
    *   `./.env:/opt/airflow/.env` (Provides environment variables to containers)
    *   *(Other mounts like `config`, `plugins`, `data` are also included)*
*   **Dependencies:** Python dependencies required by the DAGs and the underlying scripts are currently installed at container startup via the `_PIP_ADDITIONAL_REQUIREMENTS` environment variable within the `docker-compose.yaml`. See the [Dependencies](#dependencies) section.

### Running the Airflow Stack:

1.  **Navigate:** Open a terminal and `cd` to the root directory of the `PersonalTrainerAI` project (where the main `docker-compose.yaml` file resides).
2.  **Start Services:**
    ```bash
    docker-compose up -d
    ```
    *   The first time you run this, Docker will pull the specified Airflow image and then start all services.
    *   The Airflow containers (especially webserver, scheduler, worker) will then install the Python packages listed in `_PIP_ADDITIONAL_REQUIREMENTS`. **This can take several minutes.** The UI will likely be unavailable until this completes.
3.  **Access Airflow UI:** Once services are initialized, open your web browser and go to `http://localhost:8080`.
    *   **Credentials:** Log in using the default credentials defined in the `docker-compose.yaml` (`airflow-init` service section):
        *   Username: `ruthreddy`
        *   Password: `chinni`
4.  **Check Status:**
    ```bash
    docker-compose ps
    ```
5.  **View Logs:** (Essential for debugging)
    ```bash
    # View webserver logs (useful for UI issues, DAG parsing errors)
    docker-compose logs airflow-webserver

    # View scheduler logs (useful for DAG scheduling, parsing errors)
    docker-compose logs airflow-scheduler

    # View worker logs (useful for task execution errors)
    docker-compose logs airflow-worker

    # Add '-f' flag to follow logs in real-time (e.g., docker-compose logs -f airflow-worker)
    ```
6.  **Stop Services:**
    ```bash
    docker-compose down
    ```
    *(This stops and removes the containers but preserves the Postgres volume unless `--volumes` is added)*

## DAG Descriptions

### 1. Data Pipeline DAG (`Data_pipeline_dag`)

*   **File:** `data_pipeline_airflow.py`
*   **Purpose:** Orchestrates the process of collecting data from various sources, cleaning/preprocessing it, and loading it into the Pinecone vector database for the RAG models.
*   **Tasks:**
    *   `scrape_ms_task`: Scrapes data from a specific website.
    *   `preprocess_ms_task`: Preprocesses the scraped MS data.
    *   `scrape_blog_task`: Scrapes data from blogs.
    *   `scrape_pdf_task`: Scrapes data from PDF files.
    *   `preprocess_other_data_task`: Preprocesses blog and PDF data.
    *   `chunk_db_task`: Chunks the preprocessed data and upserts it into Pinecone.
*   **Trigger:** Typically triggered manually from the Airflow UI.

### 2. Model Evaluation Pipeline DAG (`model_evaluation_pipeline`)

*   **File:** `model_evaluation_pipeline_dag.py`
*   **Purpose:** Runs evaluations for both RAG implementations and the conversational Agent after the data pipeline is complete. It leverages the underlying evaluation scripts which handle MLflow logging internally.
*   **Tasks:**
    *   `wait_for_data_pipeline_completion`: An `ExternalTaskSensor` that waits for the `Data_pipeline_dag` to complete successfully before proceeding.
    *   `run_rag_evaluation`: Executes the RAG evaluation comparison logic (from `src.rag_model.advanced_rag_evaluation`), which compares different RAG setups and logs results to MLflow.
    *   `run_agent_evaluation`: Executes the Agent evaluation logic (from `src.chatbot.agent_eval.eval`), which uses LangSmith runs, a judge LLM, and logs results to MLflow.
*   **Trigger:** Automatically triggered after `Data_pipeline_dag` run succeeds. The RAG and Agent evaluation tasks run in parallel after the sensor task succeeds.

## Running the DAGs

1.  **Access UI:** Go to `http://localhost:8080` and log in.
2.  **Unpause DAGs:** Find `Data_pipeline_dag` and `model_evaluation_pipeline` in the DAGs list. Toggle them from "Paused" (Off) to "Running" (On). DAGs might take a minute or two to be parsed and appear after initial startup.
3.  **Trigger Data Pipeline:** Manually trigger the `Data_pipeline_dag` by clicking the "Play" button next to its name.
4.  **Monitor:** Observe the progress of the `Data_pipeline_dag` in the Grid or Graph view.
5.  **Automatic Evaluation:** Once the `Data_pipeline_dag` run completes successfully, the `wait_for_data_pipeline_completion` task in the `model_evaluation_pipeline` DAG will detect this and turn green. Subsequently, the `run_rag_evaluation` and `run_agent_evaluation` tasks will be scheduled and run.
6.  **Check Evaluation Results:** Monitor the evaluation tasks' logs in the Airflow UI. Check your MLflow UI (e.g., `http://localhost:5000`) for new runs logged by these tasks under their respective experiments ("rag_evaluation", "Agent Evaluation").

## Configuration (`.env` File)

The execution of the DAG tasks relies heavily on environment variables defined in the `.env` file located at the **project root** (`PersonalTrainerAI/.env`). This file is mounted into the Airflow containers. Ensure it contains **at least** the following variables with correct values:


# --- LLM / LangSmith ---
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LANGSMITH_API_KEY=ls_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LANGSMITH_PROJECT=Your_LangSmith_Project_Name # Used by Agent Eval
# LANGSMITH_TRACING=true # Optional, usually true

# --- Vector DB (Example: Pinecone) ---
PINECONE_API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
PINECONE_ENVIRONMENT=your-pinecone-environment # e.g., us-west1-gcp or similar

# --- MLflow ---
# IMPORTANT: Adjust based on where MLflow server runs relative to Docker
# If MLflow runs on HOST machine: use host.docker.internal (on Docker Desktop)
MLFLOW_TRACKING_URI=http://host.docker.internal:5000
# If MLflow runs as another container 'mlflow-server' on the same Docker network:
# MLFLOW_TRACKING_URI=http://mlflow-server:5000
# If running locally without Docker (for testing eval script directly):
# MLFLOW_TRACKING_URI=http://localhost:5000

# --- Other Optional Keys ---
# DEEPSEEK_API_KEY=sk_xxxxxxxxxxxxxxxxxxx # If judge LLM is changed back
# Add any other API keys or config needed by src code

