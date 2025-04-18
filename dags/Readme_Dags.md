# Airflow DAGs for PersonalTrainerAI

This directory contains the Apache Airflow DAG (Directed Acyclic Graph) definitions used to orchestrate the data processing and model evaluation pipelines for the PersonalTrainerAI project.

## Overview

Airflow is used here to automate and monitor the project's workflows. This ensures that data scraping, preprocessing, vector database loading, and subsequent model/agent evaluations run in the correct order and can be easily monitored.

The primary workflows orchestrated are:

1.  **Data Pipeline:** Scrapes data from various sources (websites, blogs, PDFs), preprocesses it, and loads it into a vector database (e.g., Pinecone). **Upon successful completion, this DAG automatically triggers the Model Evaluation Pipeline.**
2.  **Model Evaluation Pipeline:** Evaluates the performance of different RAG implementations and the conversational agent, logging results to MLflow. This pipeline is triggered automatically by the Data Pipeline DAG.

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
*   **Image:** It uses a specific official `apache/airflow` image tag configured for **Python 3.9 or newer** (e.g., `apache/airflow:2.9.2-python3.9`) to ensure compatibility with libraries like Langchain, MLflow, etc. This is configured in the `docker-compose.yaml` file under `x-airflow-common.image` (or via the `AIRFLOW_IMAGE_NAME` variable in `.env`).
*   **Volumes:** The following project directories are mounted into the Airflow containers, making code and configuration accessible:
    *   `./dags:/opt/airflow/dags`
    *   `./src:/opt/airflow/src`
    *   `./logs:/opt/airflow/logs`
    *   `./.env:/opt/airflow/.env`
    *   *(Other mounts like `config`, `plugins`, `data` are also included)*
*   **Environment Variables:** Critical environment variables (API keys, URIs) defined in the root `.env` file are automatically loaded into the Airflow service containers via the `docker-compose.yaml` configuration (using `${VAR_NAME}` substitution in the `environment` section). This is the **recommended way** to provide secrets and configuration to tasks.

### Dependencies

*   **Current Method:** Python package dependencies for DAG tasks are currently installed at container startup via the `_PIP_ADDITIONAL_REQUIREMENTS` environment variable in the project root `docker-compose.yaml` file. This list includes packages needed for Airflow providers, data processing, Langchain, MLflow, etc.
    *   *Drawback:* This method significantly slows down container startup time as packages are re-installed every time.
*   **Recommended Improvement (Future):** Build a custom Docker image for Airflow.
    1.  Create a dedicated `airflow_requirements.txt` containing only the necessary packages (currently listed in `_PIP_ADDITIONAL_REQUIREMENTS`).
    2.  Create a `Dockerfile` that starts `FROM apache/airflow:TAG-pythonVERSION`, copies `airflow_requirements.txt`, and runs `pip install`.
    3.  Update `docker-compose.yaml` to use `build: path/to/dockerfile/dir` instead of `image: ...`.
    4.  Run `docker-compose build` once. Subsequent `docker-compose up` commands will be much faster.

### Running the Airflow Stack:

1.  **Navigate:** Open a terminal and `cd` to the root directory of the `PersonalTrainerAI` project.
2.  **Start MLflow Server:** The MLflow server needs to be running separately for the evaluation DAGs to log results. Open a **second terminal**, navigate to the project root, activate your Python environment (optional but good), and run:
    ```bash
    # Ensure mlruns/ and mlartifacts/ directories exist
    mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri mlruns/ --artifacts-destination mlartifacts/
    ```
    *   *(Note: Port `5001` is used here to avoid conflict with macOS AirPlay on port 5000. Adjust the port and the `MLFLOW_TRACKING_URI` in `.env` if needed. Using `--host 0.0.0.0` is crucial for accessibility from Docker containers).*
    *   **Keep this MLflow server terminal running.**
3.  **Start Airflow Services:** In your first terminal (at the project root):
    ```bash
    docker-compose up -d
    ```
    *   Allow **several minutes** for initialization and package installation on first startup or after changes.
4.  **Access Airflow UI:** Open your web browser and go to `http://localhost:8080`.
    *   **Credentials:** Log in using the default credentials defined in the `docker-compose.yaml` (`airflow-init` service section), likely:
        *   Username: `ruthreddy` (or `Vinyas` based on a previous compose file edit)
        *   Password: `chinni` (or `Vinyas`)
        *(Check the `_AIRFLOW_WWW_USER_USERNAME` and `_AIRFLOW_WWW_USER_PASSWORD` defaults in your active `docker-compose.yaml`)*
5.  **Check Status:**
    ```bash
    docker-compose ps
    ```
6.  **View Logs:** (Essential for debugging)
    ```bash
    docker-compose logs airflow-webserver
    docker-compose logs airflow-scheduler
    docker-compose logs airflow-worker
    # Use 'docker-compose logs -f <service_name>' to follow logs
    ```
7.  **Stop Services:**
    ```bash
    docker-compose down
    ```

## Configuration (`.env` File)

A `.env` file at the **project root** (`PersonalTrainerAI/.env`) is essential. Ensure it contains the following variables with your actual credentials and URIs. These are automatically passed to the Airflow containers by the `docker-compose.yaml` setup.

```dotenv
# --- LLM / LangSmith ---
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LANGSMITH_API_KEY=lsv2_pt_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx # Note: Must be LANGSMITH_API_KEY
LANGSMITH_PROJECT=Your_LangSmith_Project_Name
LANGSMITH_TRACING=true # Or false

# --- Vector DB (Example: Pinecone) ---
PINECONE_API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
PINECONE_ENVIRONMENT=your-pinecone-environment # e.g., us-east-1

# --- MLflow ---
# Use host.docker.internal for Docker Desktop (Mac/Win) reaching host port 5001
# Use correct port if you changed it (e.g., 5001)
MLFLOW_TRACKING_URI=http://host.docker.internal:5001

# --- Airflow ---
# Optional: Override the image tag here if needed, MUST match Python version reqs
# AIRFLOW_IMAGE_NAME=apache/airflow:2.9.2-python3.9
AIRFLOW_UID=50000 # Use 'id -u' on Linux/Mac if needed, leave default otherwise

# --- Other ---
HEVY_API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx # If used by any script
# Add any other required environment variables