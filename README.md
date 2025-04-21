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

---

### 3. Workflow Automation — Cloud Workflows

A **GCP Workflow** is created to handle the lifecycle of the VM:

- Starts the VM programmatically.
- Can be triggered by an HTTP call or a Cloud Scheduler job.
- Provides a serverless way to orchestrate infrastructure actions.

---

### 4. Scheduling — Cloud Scheduler

To automate regular runs, a **Cloud Scheduler** job is set to trigger the workflow:

- **Schedule:** Every **Monday at 9:00 AM**.
- Ensures the pipeline runs consistently without manual intervention.

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







# Level 1 Fitness Trainer: Basic Conversational Agent

![Level 1 Agent](assets/Level1_Agent.png)

This is the foundational level of the fitness trainer agent. It's a basic conversational agent built using LangGraph that can interact with a user, provide advice, and (hypothetically) integrate with a workout tracking service ("Hevy") and a knowledge base (RAG).

## Features

*   **Basic Assessment:** Asks initial questions to understand the user's fitness goals, experience, equipment, and schedule.
*   **Knowledge Retrieval (Placeholder):** Uses a `retrieve_from_rag` function (placeholder) to simulate accessing exercise science information.
*   **Workout Planning (Placeholder):** Creates personalized workout plans and can hypothetically save them to a "Hevy" account (`tool_create_routine` - placeholder).
*   **Progress Tracking (Placeholder):**  Can fetch workout logs from "Hevy" (`tool_fetch_workouts`, `tool_get_workout_count` - placeholders) to monitor progress.
*   **Plan Adjustment (Placeholder):** Suggests modifications to routines and can hypothetically update them in "Hevy" (`tool_update_routine` - placeholder).
*   **Conversational Flow:** Uses LangGraph to manage the conversation flow, switching between the main agent logic and tool execution.
* **System Prompt**: Defines a system prompt that includes the role of the agent and instructs on how to behave with the user.

## Code Structure

*   **`AgentState`:**  A `TypedDict` to represent the agent's state, including the conversation history (`messages`) and session ID(`session_id`).
*   **`llm`:**  A `ChatOpenAI` instance using the `gpt-4o` model.
*   **`system_prompt`:**  Instructions for the LLM, defining its role and capabilities.
*   **`fitness_agent(state: AgentState) -> AgentState`:** The main agent function, handling the conversation and invoking the LLM.
*   **`graph`:** A `StateGraph` defining the workflow:
    *   `fitness_agent` node:  Handles the main interaction.
    *   `tools` node:  A `ToolNode` containing all the placeholder tool functions.
    *   Conditional edges based on tool calls (`tools_condition`).

## How it Works

1.  The user starts the conversation with a message.
2.  The `fitness_agent` node receives the message and, if it's the first interaction, prepends the `system_prompt`.
3.  The LLM is invoked with the conversation history.
4.  If the LLM generates tool calls, the `tools` node is executed (using placeholder functions).
5.  The results of the tools (or the LLM's direct response) are added to the conversation history.
6.  The process repeats until the LLM doesn't generate any more tool calls, at which point the conversation ends (or could be continued with further user input).

## Setup and Running

See the "General Notes" at the beginning of this response for installation and setup instructions.  Key steps:

1.  Install dependencies.
2.  Set up your `.env` file with your OpenAI API key.
3.  **Crucially, implement the placeholder tool functions.**  This is the core of connecting the agent to external data and services.
4.  Run the application using the `app.stream()` or `app.invoke()` methods, as shown in the general setup instructions.

## Limitations

*   **Placeholder Tools:** The most significant limitation is the reliance on placeholder functions. Without real implementations, the agent can't interact with external data or services.
*   **Simple State:** The agent's state only includes the conversation history and session ID.  It doesn't maintain a structured user profile, workout plan, or progress data.
*   **No Explicit Stages:**  The agent doesn't have distinct stages (assessment, planning, monitoring).  The `system_prompt` guides the behavior, but there's no programmatic enforcement of these stages.
*   **Basic Reasoning:**  The agent relies primarily on the LLM's inherent reasoning capabilities.  There's no separate reasoning or planning component.

This level 1 agent is a good starting point, demonstrating the basic structure of a LangGraph-based conversational agent.  However, it's very limited in its capabilities due to the placeholder tools and simple state management.


---


# Level 2 Fitness Trainer: Stage-Based Agent

![Level 2 Agent](assets/Level2_Agent.png)

This level introduces a stage-based architecture to the fitness trainer agent. It divides the interaction into distinct phases: assessment, planning, and monitoring.  This allows for a more structured and controlled interaction flow.

## Key Improvements Over Level 1

*   **Explicit Stages:** The agent now operates in three distinct stages:
    *   **`assessment`:**  Gathers user information (goals, experience, etc.).
    *   **`planning`:** Creates a workout plan based on the user profile.
    *   **`monitoring`:** Tracks progress and suggests adjustments.
*   **`AgentState` Enhancement:**  The state now includes:
    *   `user_profile`: A dictionary to store user information (still basic extraction).
    *   `workout_plan`: A dictionary to store the workout plan.
    *   `progress_data`:  A dictionary (placeholder for actual progress data).
    *   `stage`:  A literal indicating the current stage (`"assessment"`, `"planning"`, or `"monitoring"`).
*   **Stage-Specific Prompts:**  Each stage has its own `SystemMessage` prompt, providing more focused instructions to the LLM.
*   **`stage_router` Function:**  Determines the next stage based on the current state.
*   **More Structured Workflow:** The `StateGraph` now includes separate nodes for each stage (`assessment`, `planning`, `monitoring`) and uses conditional edges to transition between them.

## Code Structure

*   **`AgentState`:** Expanded to include `user_profile`, `workout_plan`, `progress_data`, and `stage`.
*   **`assessment_stage`, `planning_stage`, `monitoring_stage`:**  Separate functions for each stage, each with its own prompt.
*   **`stage_router(state: AgentState) -> str`:**  A function to determine the next stage.
*   **`workflow`:**  The `StateGraph` is updated with nodes for each stage and conditional edges to control the flow.

## How it Works

1.  The agent starts in the `assessment` stage.
2.  The `assessment_stage` function asks questions to gather user information.  (Basic profile extraction is mentioned, but a robust implementation is still needed.)
3.  The `stage_router` determines when to move to the `planning` stage (based on whether a `user_profile` exists).
4.  The `planning_stage` creates a workout plan.
5.  The `stage_router` moves the agent to the `monitoring` stage.
6.  The `monitoring_stage` (hypothetically) analyzes workout data and suggests adjustments.
7.  The agent can loop back to `planning` for major updates or end the conversation.
8.  Tool usage is integrated into each stage using conditional edges.

## Limitations

*   **Placeholder Tools and Data:**  The agent still relies heavily on placeholder tool functions and doesn't have robust data handling for `progress_data`.
*   **Simple Profile Extraction:** The profile extraction within the `assessment_stage` is mentioned but not fully implemented.
*   **Limited Reasoning:**  The agent primarily uses the LLM's reasoning within each stage. There's no separate, overarching reasoning component.
*   **Basic Plan Representation:**  The `workout_plan` is stored as a simple dictionary.  A more structured representation would be beneficial.

## Advantages over Level 1

*   **More Organized Interaction:** The stage-based approach provides a clearer structure for the conversation.
*   **Better Control:**  The `stage_router` allows for more precise control over the agent's flow.
*   **More Focused Prompts:**  Stage-specific prompts improve the LLM's performance within each phase.
*   **Improved State Management:** The `AgentState` is expanded to store more relevant information.

Level 2 represents a significant improvement over Level 1 by introducing a structured, stage-based approach.  However, it still relies on placeholder tools and lacks sophisticated reasoning and data handling.


---

# Level 3 Fitness Trainer: Orchestrator and Specialized Workers

![Level 3 Agent](assets/Level3_Agent.png)

This level introduces a more sophisticated architecture with an orchestrator node and specialized worker nodes.  This allows for more flexible and dynamic behavior, delegating tasks to the appropriate agent based on the current context.

## Key Improvements Over Level 2

*   **Orchestrator Node (`orchestrator_node`):**  A central coordinator that determines the next action based on the conversation history and user needs.  This replaces the simpler `stage_router`.
*   **Specialized Worker Nodes:**  Separate nodes for specific tasks:
    *   `assessment_worker`:  Gathers user information.
    *   `research_worker`:  Retrieves exercise science information (using the `retrieve_from_rag` placeholder).
    *   `planning_worker`:  Creates workout plans.
    *   `analysis_worker`:  Analyzes workout data (using placeholder tools).
    *   `adjustment_worker`:  Modifies workout plans.
*   **`next_action` in `AgentState`:**  The state now includes a `next_action` field (a literal) to indicate the next node to execute.
*   **Context Sharing (`context` in `AgentState`):**  A `context` dictionary is added to the state to share information between nodes (e.g., research findings, orchestrator reasoning).
*   **More Dynamic Workflow:** The orchestrator can dynamically choose the next action, allowing for more flexible transitions between tasks.

## Code Structure

*   **`AgentState`:**  Includes `next_action` and `context`.
*   **`orchestrator_node(state: AgentState) -> AgentState`:**  The orchestrator function.
*   **`assessment_worker`, `research_worker`, `planning_worker`, `analysis_worker`, `adjustment_worker`:**  Functions for each specialized worker node.
*   **`action_router(state: AgentState) -> str`:**  A simple router that returns the value of `next_action`.
*   **`workflow`:**  The `StateGraph` is significantly expanded with nodes for the orchestrator and all worker nodes.  Conditional edges are used extensively to manage the flow.

## How it Works

1.  The agent starts at the `orchestrator` node.
2.  The `orchestrator_node` analyzes the current state and determines the `next_action`.
3.  The `action_router` directs the flow to the appropriate worker node based on `next_action`.
4.  The worker node performs its task (e.g., gathering information, creating a plan, analyzing data).
5.  The worker node returns to the `orchestrator`.
6.  The process repeats, with the orchestrator dynamically choosing the next action.
7.  Tool usage is integrated into the worker nodes.
8. Orchestrator provides it reasoning in the `context` object.

## Advantages Over Level 2

*   **More Flexible and Dynamic:** The orchestrator allows the agent to adapt to the user's needs in a more dynamic way than the fixed stages of Level 2.
*   **Task Specialization:**  The worker nodes allow for more focused prompts and potentially better performance on specific tasks.
*   **Improved Context Management:**  The `context` dictionary facilitates sharing information between nodes.
*   **More Robust Decision-Making:**  The orchestrator's reasoning is more explicit than the simple `stage_router` of Level 2.

## Limitations

*   **Placeholder Tools:** The agent still heavily relies on placeholder functions for external interactions.
*   **Simple Action Parsing:** The orchestrator's action parsing is basic (string matching).  A more robust approach would be beneficial.
*   **Limited Context:** The `context` dictionary is a step forward, but a more sophisticated memory mechanism would be needed for a truly robust system.
* **Limited reasoning**: Agent still depends on the prompt for most of its reasoning.

Level 3 represents a significant architectural improvement, introducing a more flexible and dynamic multi-agent approach. However, it still has limitations related to placeholder tools, simple action parsing, and limited context management.


---

# Level 4 Fitness Trainer: Multi-Agent System with Memory and Reasoning

![Level 4 Agent](assets/Level4_Agent.png)

This level introduces a significantly more complex architecture, incorporating a multi-agent system with dedicated components for memory management, reasoning, and user modeling.  It aims to create a more intelligent and adaptable fitness trainer.

## Key Improvements Over Level 3

*   **Sophisticated State Management:**
    *   `memory`:  Long-term memory storage.
    *   `working_memory`:  Short-term contextual memory.
    *   `user_model`:  A comprehensive model of the user.
    *   `fitness_plan`:  A structured representation of the fitness plan.
    *   `reasoning_trace`:  A log of reasoning steps.
*   **Specialized Agents:** A more comprehensive set of specialized agents:
    *   `coordinator_agent`: Orchestrates the multi-agent system.
    *   `profiler_agent`:  In-depth user assessment.
    *   `research_agent`:  Fitness knowledge retrieval.
    *   `planner_agent`:  Workout plan creation.
    *   `analyst_agent`:  Progress analysis.
    *   `adaptation_agent`:  Plan adaptation.
    *   `coach_agent`:  Motivational coaching.
*   **Memory Manager (`memory_manager`):**  Manages long-term and working memory.
*   **Reasoning Engine (`reasoning_engine`):**  Performs advanced reasoning to determine the optimal next steps.
*   **User Modeler (`user_modeler`):**  Builds and maintains the `user_model`.
*   **Agent Router (`agent_router`):**  Routes based on the `current_agent` determined by the reasoning engine.
*   **Agent States (`agent_state`):** Tracks the current state of each specialized agent.

## Code Structure

*   **`AgentState`:** Significantly expanded to include `memory`, `working_memory`, `user_model`, `fitness_plan`, `reasoning_trace`, `agent_state`, and `current_agent`.
*   **`memory_manager`, `reasoning_engine`, `user_modeler`:**  Functions for managing memory, reasoning, and user modeling.
*   **`coordinator_agent`:**  The central coordinator agent.
*   **`profiler_agent`, `research_agent`, `planner_agent`, `analyst_agent`, `adaptation_agent`, `coach_agent`:**  Specialized agent functions.
*   **`agent_router(state: AgentState) -> str`:** Routes based on `current_agent`.
*   **`workflow`:**  A complex `StateGraph` connecting all the components.

## How it Works

1.  The interaction starts with the `memory_manager`.
2.  The `user_modeler` updates the user model.
3.  The `reasoning_engine` analyzes the current situation and determines the next agent to activate (`current_agent`).
4.  The `agent_router` directs the flow to the selected agent.
5.  The specialized agent performs its task.
6.  Control returns to the `coordinator`.
7.  The `coordinator` might respond to the user directly or initiate another cycle.
8.  The process continues, with the `memory_manager` and `reasoning_engine` playing key roles in guiding the interaction.

## Advantages Over Level 3

*   **Significantly Improved Reasoning:** The dedicated `reasoning_engine` allows for more sophisticated decision-making.
*   **Comprehensive Memory:**  The `memory_manager` and the various memory components (`memory`, `working_memory`, `user_model`) provide a much richer context for the agent.
*   **More Specialized Agents:** The expanded set of specialized agents allows for more focused expertise.
*   **User Modeling:** The `user_modeler` provides a more comprehensive understanding of the user.
*   **Traceability:**  The `reasoning_trace` allows for better understanding of the agent's decision-making process.

## Limitations

*   **Placeholder Tools:**  The agent still relies on placeholder functions for external interactions.
*   **Simplified Memory Updates:** The memory updates in the provided code are simplified.  A real-world system would require more sophisticated parsing and data management.
*   **Complexity:**  The architecture is significantly more complex than previous levels, making it more challenging to understand and maintain.
*   **Limited Error Handling:** Error Handling is still basic.

Level 4 is a major step forward, introducing a much more sophisticated multi-agent architecture with dedicated components for memory, reasoning, and user modeling.  This allows for a more intelligent and adaptable fitness trainer, although it still has limitations related to placeholder tools and simplified memory updates.


---

# Level 5 Fitness Trainer: Cognitive Architecture

![Level 5 Agent](assets/Level5_Agent.png)

This level implements a full cognitive architecture inspired by models like SOAR and ACT-R. It features a hierarchical controller that guides the agent through a cognitive cycle of perception, interpretation, planning, execution, monitoring, reflection, and adaptation.  This is the most advanced and complex level.

## Key Improvements Over Level 4

*   **Cognitive Cycle:** The agent operates within a defined cognitive cycle:
    *   **Perception:**  Processes new inputs.
    *   **Interpretation:**  Interprets perceptions in context.
    *   **Planning:**  Develops multi-level plans.
    *   **Execution:**  Interacts with the user or environment.
    *   **Monitoring:**  Tracks execution and detects errors.
    *   **Reflection:**  Evaluates performance and identifies improvements.
    *   **Adaptation:**  Adapts behavior, knowledge, or plans.
*   **Hierarchical Controller:**  The `controller_state` variable and the `controller_router` function manage the flow through the cognitive cycle.
*   **Error Recovery (`error_recovery_node`):**  A dedicated node for handling detected errors.
*   **Advanced State Model:**
    *   `episodic_memory`:  Memory of interactions and events.
    *   `semantic_memory`:  Conceptual knowledge and facts.
    *   `working_memory`:  Active short-term memory.
    *   `fitness_domain`:  Domain-specific knowledge.
    *   `metacognition`:  System's awareness of its own state.
    *   `current_plan`:  Current workout or interaction plan.
    *   `execution_trace`:  Record of actions and outcomes.
    *   `reflection_log`:  Self-evaluation notes.
    * `human_feedback`: captures users feedbacks.
*    **Error State**: tracks errors
* **More Robust Working Memory Updates**: uses `json.dumps` for a more robust update of the working memory.

## Code Structure

*   **`AgentState`:**  The most comprehensive state model, including all the components listed above.
*   **`CONTROLLER_STATES`:**  A literal defining the possible states of the hierarchical controller.
*   **`perception_node`, `interpretation_node`, `planning_node`, `execution_node`, `monitoring_node`, `reflection_node`, `adaptation_node`, `error_recovery_node`:**  Functions for each stage of the cognitive cycle.
*   **`controller_router(state: AgentState) -> str`:**  Routes based on `controller_state`.
*   **`workflow`:** A `StateGraph` representing the cognitive cycle and tool usage.

## How it Works

1.  The agent starts in the `perception` node.
2.  The agent progresses through the cognitive cycle (perception, interpretation, planning, execution, monitoring, reflection, adaptation) under the control of the `controller_state` and `controller_router`.
3.  Each node performs its specific task, updating the state accordingly.
4.  The `monitoring_node` detects errors, triggering the `error_recovery_node`.
5.  The `reflection_node` evaluates performance and identifies areas for improvement.
6.  The `adaptation_node` makes changes to the system based on reflection.
7.  The cycle repeats, allowing the agent to continuously learn and adapt.

## Advantages Over Level 4

*   **Cognitive Architecture:**  The agent is based on a well-defined cognitive architecture, providing a more principled approach to building intelligent agents.
*   **Hierarchical Control:** The controller provides a clear and structured way to manage the agent's behavior.
*   **Error Handling:**  The dedicated `error_recovery_node` allows for more graceful handling of errors.
*   **Continuous Learning and Adaptation:**  The reflection and adaptation mechanisms enable the agent to improve over time.
*   **More Complete State Model:** The state model includes a wider range of components, providing a richer representation of the agent's internal state.

## Limitations

*   **Placeholder Tools:**  The agent still relies on placeholder functions.
*   **Simplified Memory and Knowledge:**  While the state model is comprehensive, the actual implementation of memory and knowledge representation is still simplified.
*   **Complexity:**  This is the most complex level, requiring a deep understanding of the cognitive architecture and the interactions between the different components.
*   **Computational Cost:**  The cognitive cycle and the extensive use of the LLM can be computationally expensive.
*   **Limited Error Handling:** Even though there is an `error_recovery_node`, error handling could be improved.

Level 5 represents the most advanced and sophisticated agent, based on a cognitive architecture. This provides a framework for building a highly intelligent and adaptable fitness trainer, but it also introduces significant complexity and computational cost.


---

# Comparative Analysis of Agent Levels

| Feature                     | Level 1                   | Level 2                     | Level 3                                    | Level 4                                                                | Level 5                                                                                     |
| --------------------------- | ------------------------- | --------------------------- | ------------------------------------------ | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **Architecture**            | Basic Conversational      | Stage-Based                 | Orchestrator and Workers                   | Multi-Agent System with Memory and Reasoning                           | Cognitive Architecture                                                                          |
| **State Management**       | Minimal (messages, session_id)        | Stage, User Profile, Plan | + `next_action`, `context`                | + Long/Short-Term Memory, User Model, Reasoning Trace, Agent States | + Episodic/Semantic Memory, Metacognition, Execution Trace, Reflection Log, Error State, Human Feedback                 |
| **Decision-Making**        | LLM-Driven (System Prompt) | Stage Router               | Orchestrator (Simple Parsing)              | Reasoning Engine (Advanced)                                            | Hierarchical Controller (Cognitive Cycle)                                                     |
| **Flexibility**             | Low                       | Moderate                    | High                                       | Very High                                                              | Highest                                                                                     |
| **Specialization**         | None                      | Basic Stages                | Specialized Worker Nodes                   | Comprehensive Specialized Agents                                       | Cognitive Components (Perception, Interpretation, etc.)                                 |
| **Memory**                  | None                      | Limited                     | Context Sharing                           | Long-Term, Short-Term, User Model                                       | Episodic, Semantic, Working, User Model, Fitness Domain, Metacognition                     |
| **Reasoning**               | Basic (LLM-Inherent)      | Basic (Stage-Specific)      | Orchestrator-Based                         | Dedicated Reasoning Engine                                             | Integrated into Cognitive Cycle                                                              |
| **Adaptation**             | Limited (System Prompt)   | Limited (Stage Transitions) | Limited (Orchestrator-Driven)                | Plan Adaptation Agent                                                 | Reflection and Adaptation Nodes                                                              |
| **Error Handling**        | None                      | None                     |  None                   | None                                        | Dedicated Error Recovery Node                                                                 |
| **Complexity**             | Low                       | Moderate                    | Moderate-High                              | High                                                                  | Very High                                                                                   |
| **Key Components**          | `fitness_agent`, `tools`   | Stage Functions, `stage_router` | Orchestrator, Worker Nodes, `action_router` | Memory Manager, Reasoning Engine, User Modeler, Specialized Agents    | Cognitive Cycle Nodes, Controller, Comprehensive State Model                            |

**Summary of Progression:**

*   **Level 1:**  Foundation - basic conversation and placeholder tools.
*   **Level 2:**  Structure - introduces stages for a more organized interaction.
*   **Level 3:**  Flexibility - orchestrator and workers for dynamic task delegation.
*   **Level 4:**  Intelligence - multi-agent system with memory, reasoning, and user modeling.
*   **Level 5:**  Cognition - cognitive architecture for a principled, adaptive, and self-aware agent.

Each level builds upon the previous one, adding complexity and capabilities. The choice of which level to use depends on the specific requirements of the application, balancing the need for intelligence and adaptability with the constraints of development time, computational resources, and maintainability. Level 5, while the most sophisticated, is also the most complex and resource-intensive. Level 1, while simple, might be sufficient for very basic interactions. Levels 2-4 offer intermediate options, providing increasing levels of sophistication. The crucial next step for *any* of these levels is to replace the placeholder tool functions with real implementations.


