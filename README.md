# Personal Trainer AI

The PersonalTrainerAI project aims to revolutionize personal fitness training by leveraging artificial intelligence. This project will develop an AI-powered personal trainer that provides customized workout plans, real-time feedback, and progress tracking to users. By utilizing advanced machine learning algorithms, the AI trainer will adapt to individual fitness levels and goals, ensuring a personalized and effective training experience.

The primary objective of this project is to make personal training accessible and affordable for everyone. With the AI trainer, users can receive professional guidance and support without the need for expensive gym memberships or personal trainers. This solution is designed to promote a healthier lifestyle and help users achieve their fitness goals efficiently.

Our project includes developing a comprehensive Machine Learning Operations (MLOps) pipeline, encompassing data collection, preprocessing, model training, and deployment. The AI trainer will be available as a user-friendly mobile application, allowing users to conveniently access their personalized workout plans and track their progress anytime, anywhere.

## Table of Contents

1. [Directory Structure](#directory-structure)
2. [Setup and Usage](#setup-and-usage)
3. [Environment Setup](#environment-setup)
4. [Running the Pipeline](#running-the-pipeline)
5. [Test Functions](#test-functions)
6. [Reproducibility and Data Versioning](#reproducibility-and-data-versioning)
7. [Data Sources](#data-sources)
8. [Schema](#schema)
9. [Bias Detection](#bias-detection)
10. [Airflow Implementation](#airflow-implementation)
11. [Pipeline Tasks Overview](#pipeline-tasks-overview)

## Directory Structure

## 📂 Project Directory Structure

```
.
├── .dvc/
│   ├── .gitignore
│   └── config
│
├── .github/
│   └── workflows/
│       ├── data_pipeline_to_gcp.yml
│       └── python-tests.yml
│
├── alembic/
│   ├── README
│   ├── env.py
│   ├── script.py.mako
│
├── dags/
│   └── data_pipeline_airflow.py
│
├── data/
│   ├── preprocessed_json_data/
│   │   ├── .gitignore
│   │   ├── blogs.json.dvc
│   │   ├── ms_data.json.dvc
│   │   └── pdf_data.json.dvc
│   └── raw_json_data/
│       ├── .gitignore
│       ├── blogs.json.dvc
│       ├── ms_data.json.dvc
│       └── pdf_data.json.dvc
│
├── experiments/
│   └── pranav/
│       └── agent/
│           ├── assets/
│           ├── README.md
│           ├── __init__.py
│           ├── agent.ipynb
│           ├── basic_agent.py
│           ├── cognitive_agent.py
│           ├── multi_agent.py
│           ├── new_agent_architecture.py
│           ├── orchestrator_worker_agent.py
│           └── stage_based_agent.py
│
├── logs/
│   ├── preprocessing.log
│   ├── scraper.log
│   └── vectordb.log
│
├── logstash/
│   ├── config/
│   │   ├── logstash.yml
│   │   └── pipelines.yml
│   └── pipeline/
│       ├── fitness-chatbot.conf
│       └── minimal.conf
│
├── result/
│   ├── advanced_evaluation_results.json
│   ├── fitness_domain_metrics_comparison.png
│   ├── human_evaluation_metrics_comparison.png
│   ├── overall_comparison.png
│   ├── ragas_metrics_comparison.png
│   ├── response_time_comparison.png
│   └── retrieval_metrics_comparison.png
│
├── results/
│   ├── evaluation_results.json
│   ├── metrics_comparison.png
│   └── response_time_comparison.png
│
├── src/
│   ├── chatbot/
│   │   └── agent/
│   │       ├── __init__.py
│   │       ├── agent_models.py
│   │       ├── graph.py
│   │       ├── hevy_api.py
│   │       ├── hevy_exercises.json
│   │       ├── llm_tools.py
│   │       ├── personal_trainer_agent.py
│   │       ├── prompts.py
│   │       ├── test_api.py
│   │       └── utils.py
│   ├── agent_eval/
│   │   └── eval.py
│   ├── alembic/
│   │   └── versions/
│   ├── README.md
│   ├── __init__.py
│   ├── alembic.ini
│   ├── chat_client.py
│   ├── config.py
│   ├── elk_logging.py
│   ├── experiments.ipynb
│   ├── hevy_exercises.json
│   ├── main.py
│   ├── models.py
│   └── redis_utils.py
│
├── data_pipeline/
│   ├── __init__.py
│   ├── blogs.py
│   ├── ms.py
│   ├── ms_preprocess.py
│   ├── other_preprocesing.py
│   ├── pdfs.py
│   └── vector_db.py
│
├── other/
│   ├── __init__.py
│   └── bias_detection.py
│
├── rag_model/
│   ├── .DS_Store
│   └── __init__.py
│
├── tests/
│   ├── advanced_rag_evaluation_test.py
│   ├── advanced_rag_test.py
│   ├── modular_rag_test.py
│   ├── raptor_rag_test.py
│   ├── test_ms.py
│   ├── test_ms_preprocess.py
│   ├── test_other_preprocessing.py
│   ├── test_pdf_scraper.py
│   └── test_vectdb.py
│
├── .dockerignore
├── .dvcignore
├── .env.example
├── .env.local
├── .gitignore
├── 2.8.0
├── Dockerfile
├── Dockerfile.frontend
├── Dockerfile.test
├── ELK-INTEGRATION.md
├── README.md
├── Scoping.md
├── alembic.ini
├── docker-compose-elk.yml
├── docker-compose.chatbot.yml
├── docker-compose.local.yml
├── docker-compose.yaml
├── image.png
├── kibana-dashboard-setup.md
└── requirements.frontend.txt
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

## Data Sources

- Muscle and Strength workout data: https://www.muscleandstrength.com/workouts/men
- Articles are scraped from the link :
- "https://www.muscleandstrength.com/articles/beginners-guide-to-zone-2-cardio",
    "https://www.muscleandstrength.com/articles/best-hiit-routines-gym-equipment",
    "https://jeffnippard.com/blogs/news",
    "https://rpstrength.com/blogs/articles",
    "https://rpstrength.com/collections/guides",
    "https://www.strongerbyscience.com/complete-strength-training-guide/",
    "https://www.strongerbyscience.com/how-to-squat/",
    "https://www.strongerbyscience.com/how-to-bench/",
    "https://www.strongerbyscience.com/how-to-deadlift/",
    "https://www.strongerbyscience.com/hypertrophy-range-fact-fiction/",
    "https://www.strongerbyscience.com/metabolic-adaptation/"
- Pdfs are downloaded from different websites and stored locally

## Schema

- **Source**: Website where the workout was retrieved.
- **Title**: Workout program name.
- **URL**: Link to the full workout details.
- **Description**: Description of the workout with summary.

## Bias Detection

### Overview
This module analyzes potential biases in the PersonalTrainerAI dataset, specifically focusing on gender representation and workout intensity distribution. It helps identify imbalances in fitness recommendations and ensures fairness in chatbot responses.

### How It Works
1. **Load Data**: Reads workout descriptions from JSON files.
2. **Gender Bias Analysis**: Identifies if workouts are male, female, or unisex-focused.
3. **Workout Intensity Analysis**: Classifies workouts as High, Medium, or Low intensity.
4. **Bias Evaluation**: Uses Fairlearn to measure accuracy differences in recommendations across gender groups.

### Findings
- Initially, the data was biased towards men and unisex workouts.
- More balanced data was added to ensure fairness.
- Now, workout recommendations are more evenly distributed.

### Running Bias Detection
To check for bias, run:

```bash
python bias_detection.py
```

It will analyze `pdf_data.json`, `ms_data.json`, and `blogs.json`.



## Airflow Implementation

![Airflow Implementation](image.png)

## Pipeline Tasks Overview

### Scraping Tasks (Extract Data)

#### `scrape_ms_task` ( Success)
- **Description:** Scrapes data from Muscle & Strength (MS) website.
- **Details:** Fetches workout plans and fitness-related content.
- **Implementation:** Uses a `PythonOperator` to execute the scraping logic.

#### `scrape_blog_task` ( Success)
- **Description:** Scrapes blogs and articles from various fitness sources.
- **Details:** Extracts information from fitness blogs, research, and expert articles.

#### `scrape_pdf_task` (Success)
- **Description:** Extracts text from fitness-related PDFs.
- **Implementation:** Uses `PyPDF2` or another PDF parsing tool to retrieve content.

###  Preprocessing Tasks (Transform Data)

#### `preprocess_ms_task` ( Success)
- **Description:** Cleans and preprocesses the raw data from the Muscle & Strength website.
- **Details:** Formats the data into a structured JSON format.

#### `preprocess_other_data_task` ( Success)
- **Description:** Merges and preprocesses data from blogs and PDFs.
- **Details:** Ensures the data is structured and cleaned before vectorization.

###  Vectorization and Database Storage (Load Data)

#### `chunk_db_task` ( Success)
- **Description:** Converts text data into vector embeddings.
- **Implementation:** Uses Pinecone or another vector database to store the embeddings.
- **Details:** Enables RAG (Retrieval-Augmented Generation) to query fitness content efficiently.

### DAG Execution Flow
- `scrape_ms_task` → `preprocess_ms_task`
- `scrape_blog_task` & `scrape_pdf_task` → `preprocess_other_data_task`
- `preprocess_ms_task` & `preprocess_other_data_task` → `chunk_db_task`

## Test Functions

###  `test_ms.py`
** Test Cases**
- **Test Successful Scraping:** Ensures workout title, URL, and description are extracted.
- **Test Handling of Missing Fields:** Checks how scraper reacts to empty/missing fields.
- **Test URL Validity:** Verifies that extracted workout URLs are correct and accessible.

###  `test_ms_preprocess.py`
**Test Cases**
- **Test Text Cleaning:** Ensures removal of HTML tags, special characters, and extra spaces.
- **Test Summary Formatting:** Verifies if the summary fields are structured correctly.
- **Test Consistent Output:** Checks if preprocessed data maintains expected structure.

###  `test_other_preprocessing.py`
**Test Cases**
- **Test Blog Cleaning:** Ensures extracted text is clean and readable.
- **Test Article Formatting:** Checks if article headers, subheaders, and content are well-structured.
- **Test Handling of Noisy Data:** Ensures that unwanted elements like ads, pop-ups, or scripts are removed.

###  `test_pdf_scraper.py`
**Test Cases**
- **Test PDF Extraction:** Ensures text is extracted from multiple pages correctly.
- **Test Handling of Non-Text Elements:** Verifies that tables/images do not break extraction.
- **Test Empty or Corrupted PDFs:** Ensures the function gracefully handles unreadable PDFs.

### `test_vectdb.py`
**Test Cases**
- **Test Embedding Generation:** Ensures vector embeddings are correctly computed.
- **Test Storage in Pinecone:** Verifies that embeddings are successfully stored in the database.
- **Test Query Retrieval:** Ensures that similarity search returns relevant results.

## Cloud Setup: GCP Infrastructure for Data Pipeline

This project leverages Google Cloud Platform (GCP) to run a fully automated, cost-efficient data pipeline. Below is a breakdown of the cloud components and their integration.

---

### 1. Code Storage — Google Cloud Storage (GCS)

All source code is stored in a **GCS bucket**, which serves as the central repository for pipeline scripts.

- Acts as the single source of truth.
- Updated automatically via CI/CD (see section 5).
- Ensures the VM always runs the latest version of the code.

---

### 2. Execution — Google Compute Engine (VM)

The pipeline executes on a **GCP Virtual Machine** only when required.

- A **startup script** is configured on the VM to:
  - Download the latest code from the GCS bucket.
  - Run the pipeline.
  - Automatically shut down the VM upon completion.
- This ensures compute costs are minimized.

  ![image](https://github.com/user-attachments/assets/eb137c34-adae-4804-acf3-1063febe763c)


---

### 3. Workflow Automation — Cloud Workflows

A **GCP Workflow** is created to handle the lifecycle of the VM:

- Starts the VM programmatically.
- Can be triggered by an HTTP call or a Cloud Scheduler job.
- Provides a serverless way to orchestrate infrastructure actions.
![image](https://github.com/user-attachments/assets/d47ccb13-8e31-4b94-935c-3e6462c3e049)

---

### 4. Scheduling — Cloud Scheduler

To automate regular runs, a **Cloud Scheduler** job is set to trigger the workflow:

- **Schedule:** Every **Monday at 9:00 AM**.
- Ensures the pipeline runs consistently without manual intervention.

  ![image](https://github.com/user-attachments/assets/c6a8eed5-431c-48e8-84a8-003800753a38)


---

###  5. Continuous Deployment — GitHub Actions

A **GitHub Actions** workflow handles CI/CD whenever code is pushed to the main repository:

- Automatically syncs new or updated files to the GCS bucket.
- Ensures the cloud bucket always contains the most recent version of the codebase.
- Keeps deployments streamlined and version-controlled.

---

###  6. VM Startup Script Logic

The VM’s startup script performs the following tasks in order:

1. **Download latest code** from the GCS bucket.
2. **Execute pipeline scripts** (e.g., Python).
3. **Shut down the VM** to prevent unnecessary costs.

This makes the system stateless, self-cleaning, and fully automated.

---

###  Summary Table

| Component           | Purpose                                                            |
|---------------------|--------------------------------------------------------------------|
| **GCS Bucket**      | Stores and serves the latest codebase                             |
| **GCE VM**          | Executes the data pipeline                                         |
| **Cloud Workflow**  | Starts the VM and orchestrates execution                          |
| **Cloud Scheduler** | Triggers the workflow every Monday at 9:00 AM                     |
| **GitHub Actions**  | Pushes local changes to GCS bucket automatically                  |
| **Startup Script**  | Pulls code, runs the job, and shuts down the VM                   |

---

This setup ensures a **serverless, cost-effective, and continuously deployed** data pipeline with minimal manual intervention.

![flow](https://github.com/user-attachments/assets/c81475cd-3e22-4df1-a355-ed352e9e08fd)


### End

## RAG Implementation

"""
Updated README for RAG Model Implementation in PersonalTrainerAI

This document provides an overview of the RAG (Retrieval Augmented Generation) implementations
for the PersonalTrainerAI project, including the newly added Graph RAG and RAPTOR RAG architectures.
"""

# RAG Model Implementation for PersonalTrainerAI

This directory contains the implementation of five different RAG (Retrieval Augmented Generation) architectures for the PersonalTrainerAI project. These implementations allow the AI to retrieve relevant fitness knowledge from the vector database and generate accurate, helpful responses to user queries.

## Overview

The RAG model combines the power of large language models with retrieval from a domain-specific knowledge base. For PersonalTrainerAI, this means retrieving fitness knowledge from our Pinecone vector database and using it to generate personalized fitness advice.

We've implemented five different RAG architectures to compare their performance:

1. **Naive RAG**: A baseline implementation with direct vector similarity search
2. **Advanced RAG**: Enhanced with query expansion, sentence-window retrieval, and re-ranking
3. **Modular RAG**: A flexible system with query classification and specialized retrievers
4. **Graph RAG**: Uses a knowledge graph structure to represent relationships between fitness concepts
5. **RAPTOR RAG**: Employs multi-step reasoning with iterative retrieval for complex queries

## Architecture Comparison

### Naive RAG
- Simple vector similarity search
- Direct document retrieval
- Basic prompt construction
- Good for straightforward fitness queries

### Advanced RAG
- Query expansion using LLM
- Sentence-window retrieval for better context
- Re-ranking of retrieved documents
- Dynamic context window based on relevance
- Structured prompt engineering
- Better for nuanced fitness questions

### Modular RAG
- Query classification
- Specialized retrievers for different fitness topics
- Template-based responses
- Excellent for diverse query types

### Graph RAG
- Knowledge graph construction from fitness documents
- Graph-based retrieval using node relationships
- Path-aware context augmentation
- Relationship-enhanced prompting
- Multi-hop reasoning for complex queries
- Excels at questions involving relationships between fitness concepts

### RAPTOR RAG
- Query planning and decomposition
- Iterative, multi-step retrieval
- Reasoning over retrieved information
- Self-reflection and refinement
- Structured response synthesis
- Best for complex, multi-part fitness questions

## Getting Started

### Prerequisites

- Python 3.8+
- Pinecone account with an index set up
- OpenAI API key

### Environment Setup

Create a `.env` file in the project root with:

```
# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=personal-trainer-ai

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
```

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The requirements include:
```
langchain>=0.1.0
langchain-openai>=0.0.2
langchain-community>=0.0.10
langchain-core>=0.1.0
openai>=1.0.0
pinecone-client>=2.2.1
sentence-transformers>=2.2.2
python-dotenv>=1.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.2.0
tqdm>=4.65.0
matplotlib>=3.7.0
pydantic>=2.0.0
networkx>=2.8.0  # For Graph RAG
```

## Usage

### Comparing RAG Implementations

To compare all five RAG implementations and determine which works best for your fitness knowledge base:

```bash
python -m src.rag_model.compare_rag_implementations --output-dir results
```

This will:
1. Run test queries through all five RAG implementations
2. Evaluate responses using multiple metrics
3. Generate comparison charts and a detailed report
4. Identify the best-performing implementation

Additional options:
```bash
python -m src.rag_model.compare_rag_implementations --help
```

### Using a Specific RAG Implementation

To use a specific RAG implementation for processing queries:

```bash
python -m src.rag_model.rag_integration --implementation [naive|advanced|modular|graph|raptor]
```

For example, to use the Graph RAG implementation:
```bash
python -m src.rag_model.rag_integration --implementation graph
```

### Building the Knowledge Graph (for Graph RAG)

To build and save the knowledge graph for Graph RAG:

```bash
python -m src.rag_model.graph_rag --build-graph --graph-path fitness_knowledge_graph.json
```

## Evaluation Framework

The evaluation framework in `rag_evaluation.py` assesses RAG performance using these metrics:

1. **Relevance**: How well the response addresses the user's query
2. **Factual Accuracy**: Whether the information provided is correct
3. **Completeness**: Whether the response covers all important aspects
4. **Hallucination**: Whether the response contains information not in the retrieved documents
5. **Relationship Awareness**: How well the response demonstrates understanding of relationships between fitness concepts
6. **Reasoning Quality**: The quality of reasoning demonstrated in complex responses

## Integration with Data Pipeline

The RAG implementations integrate with the existing data pipeline:

1. Data is scraped from fitness sources
2. Text is processed and chunked
3. Chunks are embedded and stored in Pinecone
4. RAG retrieves relevant chunks based on user queries
5. LLM generates responses using the retrieved information

## File Structure

- `__init__.py`: Package initialization
- `naive_rag.py`: Baseline RAG implementation
- `advanced_rag.py`: Enhanced RAG with additional techniques
- `modular_rag.py`: Modular RAG with specialized retrievers
- `graph_rag.py`: Graph-based RAG using knowledge graph
- `raptor_rag.py`: RAPTOR RAG with multi-step reasoning
- `rag_evaluation.py`: Evaluation framework for comparing implementations
- `compare_rag_implementations.py`: Script to run comparisons
- `rag_integration.py`: Integration with the existing pipeline
- `rag_implementation_strategy.md`: Detailed strategy document
- `README.md`: This documentation file

## Contributing

When extending or modifying the RAG implementations:

1. Ensure all implementations follow the same interface
2. Add appropriate evaluation metrics for new techniques
3. Update the comparison script to include new implementations
4. Document any new parameters or configuration options

## Next Steps

- Fine-tune the knowledge graph for Graph RAG with domain expert input
- Optimize RAPTOR RAG's reasoning process for better performance
- Implement hybrid approaches combining the strengths of different architectures
- Develop a feedback loop to continuously improve RAG performance based on user interactions


## Chatbot Implementation

# Chatbot Application Setup and Usage Guide

### 1. **Frontend (Streamlit)**

- **File:** `chat_client.py`
- Users enter a username and chat with the fitness assistant.
- Sessions are stored and resumed based on username.
- Chat history is retrieved from the backend via REST.

### 2. **Backend (FastAPI + LangGraph)**

- **File:** `main.py`
- REST endpoints for:
  - Creating users
  - Creating and fetching chat sessions
  - Posting and retrieving messages
- Uses:
  - **PostgreSQL** (via SQLAlchemy): permanent session/message storage
  - **Redis**: fast chat history caching
  - **LangGraph**: structured agent logic with memory and checkpointer
  - **OpenAI GPT**: generates AI responses

### 3. **Agent System**

- **Folder:** `agent/`
- Prompts and graph logic to drive the AI’s memory, summarization, and recommendation behavior.

### 4. **Data & Logging**

- **ELK Stack Integration** via `elk_logging.py`
- Fitness exercises (e.g. from Hevy) are stored in `hevy_exercises.json`
- Additional logs and experiments tracked in `experiments.ipynb`

---

## 🐳 Deployment

This module is containerized and deployed via **Cloud Run** using a `Dockerfile`. GitHub Actions triggers automated CI/CD workflows.

To run locally:

```bash
cd src/chatbot

# Install requirements
pip install -r requirements.txt

# Start backend (with uvicorn)
uvicorn main:app --reload --port 8000

# Start frontend
streamlit run chat_client.py


```
