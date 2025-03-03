# Personal Trainer AI

The PersonalTrainerAI project aims to revolutionize personal fitness training by leveraging artificial intelligence. This project will develop an AI-powered personal trainer that provides customized workout plans, real-time feedback, and progress tracking to users. By utilizing advanced machine learning algorithms, the AI trainer will adapt to individual fitness levels and goals, ensuring a personalized and effective training experience.

The primary objective of this project is to make personal training accessible and affordable for everyone. With the AI trainer, users can receive professional guidance and support without the need for expensive gym memberships or personal trainers. This solution is designed to promote a healthier lifestyle and help users achieve their fitness goals efficiently.

Our project includes developing a comprehensive Machine Learning Operations (MLOps) pipeline, encompassing data collection, preprocessing, model training, and deployment. The AI trainer will be available as a user-friendly mobile application, allowing users to conveniently access their personalized workout plans and track their progress anytime, anywhere.

## Table of Contents

1. [Directory Structure](#directory-structure)
2. [Data Sources](#data-sources)
3. [Setup and Usage](#setup-and-usage)
4. [Environment Setup](#environment-setup)
5. [Running the Pipeline](#running-the-pipeline)
6. [Test Functions](#test-functions)
7. [Reproducibility and Data Versioning](#reproducibility-and-data-versioning)
8. [Airflow Implementation](#airflow-implementation)
9. [Pipeline Components](#pipeline-components)
10. [CI/CD Prerequisites](#cicd-prerequisites)
11. [Model Serving and Deployment](#model-serving-and-deployment)
12. [Monitoring and Maintenance](#monitoring-and-maintenance)
13. [Notifications](#notifications)
14. [UI Dashboard](#ui-dashboard)

## Directory Structure

```
PersonalTrainerAI/
│── .dvc/
│   ├── cache/
│   ├── tmp/
│── .gitignore
│── config/
│── dags/
│   ├── data_pipeline_airflow.py
│── data/
│   ├── pdf_raw_json_data/
│   ├── preprocessed_json_data/
│   ├── raw_json_data/
│   ├── source_pdf/
│── logs/
│   ├── vectordb.log
│   ├── tscraper.log
│   ├── preprocessing.log
│── src/
│   ├── data_pipeline/
│   │   ├── __init__.py
│   │   ├── blogs.py
│   │   ├── ms_preprocess.py
│   │   ├── ms.py
│   │   ├── other_preprocessing.py
│   │   ├── pdfs.py
│   │   ├── vector_db.py
│   ├── other/
│   │   ├── __init__.py
│   │   ├── bias_detection.py
│── tests/
│   ├── test_ms_preprocess.py
│   ├── test_ms.py
│   ├── test_other_preprocessing.py
│   ├── test_pdf_scraper.py
│   ├── test_vectordb.py
│── .dvcignore
│── .env
│── docker-compose.yaml
│── README.md
│── requirements.txt
│── Scoping.md
```

## Setup and Usage

To set up and run the pipeline:

- Refer to [Environment Setup](#environment-setup)
- Refer to [Running the Pipeline](#running-the-pipeline)

## Environment Setup

To set up and run this project, ensure the following are installed:

- Python (3.8 or later)
- Docker (for running Apache Airflow)
- DVC (for data version control)
- Google Cloud SDK (we are deploying on GCP)

### Installation Steps

1. **Clone the Repository**

    ```sh
    git clone https://github.com/Pranavbp525/PersonalTrainerAI.git
    cd PersonalTrainerAI
    ```

2. **Install Python Dependencies**

    Install all required packages listed in `requirements.txt`:

    ```sh
    pip install -r requirements.txt
    ```

3. **Initialize DVC**

    Set up DVC to manage large data files by pulling the tracked data:

    ```sh
    dvc pull
    ```

## Running the Pipeline

To execute the data pipeline, follow these steps:

### Start Airflow Services

Run Docker Compose to start services of the Airflow web server, scheduler:

```sh
docker-compose up
```

### Access Airflow UI

Open [http://localhost:8080](http://localhost:8080) in your browser. Log into the Airflow UI and enable the DAG.

### Trigger the DAG

Trigger the DAG manually to start processing. The pipeline will:

- Scrape the data from website and preprocess it and stores the data in data/preprocessed_json_data/.
- Scrape the data from pdfs and preprocess it data/preprocessed_json_data/.
- Scrape the data from blogs and preprocess it data/preprocessed_json_data/.
- In the next steps the data is chunked and stored into pinecone.


### Check Outputs

Once completed, check the output files in data/preprocessed_json_data/ and the chunked data that is stored in pinecone.

Alternatively, you can follow these steps:

1. **Activate virtual environment:**

    ```sh
    python -m venv airflow_env
    source bin/activate
    ```

2. **Install Airflow (not required if done before):**

    ```sh
    pip install apache-airflow
    ```

3. **Initialize Airflow database (not required if done before):**

    ```sh
    airflow db init
    ```

4. **Start Airflow web server and scheduler:**

    ```sh
    airflow webserver -p 8080 & airflow scheduler
    ```

5. **Access Airflow UI in your default browser:**

    [http://localhost:8080](http://localhost:8080)

6. **Deactivate virtual environment (after work completion):**

    ```sh
    deactivate
    ```

## Test Functions

Run all tests in the tests directory:

```sh
pytest tests/
```

## Reproducibility and Data Versioning

We used DVC (Data Version Control) for files management.

### DVC Setup

Initialize DVC (not required if already initialized):

```sh
dvc init
```

### Pull Data Files

Pull the DVC-tracked data files to ensure all required datasets are available:

```sh
dvc pull
```

### Data Versioning

Data files are generated with .dvc files in the repository.

### Tracking New Data

If new files are added, to track them. Example:

```sh
dvc add <file-path>
dvc push
```