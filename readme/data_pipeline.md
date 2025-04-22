# Data PIPELINE

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

![image](https://github.com/user-attachments/assets/5b9efcff-4781-4024-bfd5-8373d75e3de0)


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

![image](https://github.com/user-attachments/assets/db58bda4-7127-407f-9da1-486d904063a5)



### End
