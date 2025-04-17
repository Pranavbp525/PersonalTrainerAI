# Evaluation Pipeline

This project includes pipelines for evaluating both the RAG (Retrieval-Augmented Generation) components and the Agent's performance. Evaluations are tracked using MLflow.

## Agent Evaluation with LangSmith & MLflow Tracking

This section details how to run the agent evaluation script (`src/chatbot/agent_eval/eval.py`), which assesses the quality of recorded LLM runs from a LangSmith project using a separate "judge" LLM and logs the results to MLflow.

### Overview

The script performs the following steps:

1.  Fetches recent LLM runs from a specified LangSmith project.
2.  For each run, extracts the prompt and the LLM's completion.
3.  Uses a designated "judge" LLM (e.g., DeepSeek, GPT-4) to evaluate the completion based on the prompt according to predefined criteria (Instruction Following, Relevance, Coherence).
4.  The judge LLM provides a score (0-100) and reasoning for each evaluated run.
5.  Posts this feedback (score and reasoning) back to the corresponding run in LangSmith.
6.  Calculates the average quality score across all successfully evaluated runs.
7.  Logs an aggregated evaluation run to MLflow, including:
    *   **Parameters:** Judge LLM model, LangSmith project details, number of runs evaluated, etc.
    *   **Metrics:** Average quality score.
    *   **Artifacts:** A JSON file containing the individual judgements (run ID, score, reasoning) for each evaluated LangSmith run.

### Prerequisites

1.  **Python Environment:** Ensure you have a working Python environment (virtual environment recommended) with all necessary packages installed. You can install them using:
    ```bash
    # Activate your virtual environment first!
    # Option A: Install all project dependencies
    pip install -r requirements.txt
    # Option B: Install core dependencies for this script (if Option A fails)
    # pip install python-dotenv langsmith openai pydantic mlflow deepseek langchain scikit-learn
    ```
2.  **`.env` File:** A `.env` file must exist in the project root (`PersonalTrainerAI/.env`) containing the following environment variables:
    *   `LANGSMITH_API_KEY`: Your LangSmith API key.
    *   `LANGSMITH_PROJECT`: The name of your LangSmith project containing the runs to evaluate.
    *   `LANGSMITH_TRACING`: Typically `true`.
    *   `DEEPSEEK_API_KEY` (or key for your chosen judge LLM): API key for the judge LLM defined in the script (`JUDGE_LLM_MODEL`).
    *   `MLFLOW_TRACKING_URI`: The URI of your MLflow tracking server (e.g., `http://localhost:5000`).
    *   `OPENAI_API_KEY`: May be needed by underlying Langchain components even if not the judge.
    *   *(Ensure this `.env` file is included in your `.gitignore`)*
3.  **LangSmith Project Runs:** The specified `LANGSMITH_PROJECT` must contain recent LLM runs suitable for evaluation.
4.  **Running MLflow Server:** The MLflow tracking server must be running and accessible at the `MLFLOW_TRACKING_URI`. To start a local server:
    ```bash
    # In a separate terminal, navigate to the project root and activate venv
    python src/rag_model/mlflow/mlflow_rag_tracker.py
    # OR use the direct command:
    # mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri mlruns --artifacts-destination mlartifacts
    ```
    Keep the MLflow server terminal running.

### Configuration

The evaluation script uses the following configurations (primarily set via environment variables loaded from `.env` or constants within the script):

*   **`LANGSMITH_PROJECT`**: Target LangSmith project (from `.env`).
*   **`JUDGE_LLM_MODEL`**: Model used for judging (currently set as a constant in `eval.py`, e.g., `"deepseek-chat"`).
*   **Judge LLM API Key**: Corresponding API key for the judge model (from `.env`).
*   **`MLFLOW_TRACKING_URI`**: MLflow server location (from `.env`).
*   **`AGENT_MLFLOW_EXPERIMENT_NAME`**: Name of the MLflow experiment (currently set as a constant in `eval.py`, e.g., `"Agent Evaluation"`).
*   **`FETCH_RUNS_LIMIT`**: Maximum number of runs to fetch from LangSmith (currently set as a constant in `eval.py`).
*   **`FEEDBACK_KEY`**: The key used when posting feedback to LangSmith (currently generated with date in `eval.py`).

### Running the Evaluation

1.  Ensure all prerequisites are met (dependencies installed, `.env` configured, MLflow server running).
2.  Open your terminal and navigate to the project root directory (`PersonalTrainerAI/`).
3.  Activate your virtual environment.
4.  Run the script using the Python `-m` flag:
    ```bash
    python -m src.chatbot.agent_eval.eval
    ```

### Output and Verification

1.  **Console:** Monitor the script's output for logs indicating fetching, judging, feedback posting, average score calculation, and MLflow logging status. Check for any errors or warnings.
2.  **MLflow UI:**
    *   Navigate to your MLflow tracking URI (e.g., `http://localhost:5000`).
    *   Find the experiment named **`Agent Evaluation`**.
    *   Look for the latest run (named like `AgentEval_llm_run_quality_score_...`).
    *   Inspect the run's **Parameters**, **Metrics** (specifically `avg_llm_quality_score`), and **Artifacts** (check for `llm_run_judgements.json`).
3.  **LangSmith UI:**
    *   Go to your LangSmith project online.
    *   Select some runs that were processed by the script.
    *   Go to the "Feedback" tab for those runs.
    *   You should see feedback entries with the key used by the script (e.g., `llm_run_quality_score_YYYYMMDD`), the normalized score (0.0-1.0), and the judge's reasoning in the comment.

### Troubleshooting

*   **`No module named 'src'` / `ImportError`:** Ensure you are running the script from the project root directory using the `python -m src.chatbot.agent_eval.eval` command. Check virtual environment activation.
*   **`Connection Refused [WinError 10061]` to MLflow:** The MLflow tracking server is likely not running or not accessible at the specified `MLFLOW_TRACKING_URI`. Start the server using the command in the prerequisites.
*   **Environment Variable Errors:** Double-check that the `.env` file exists in the project root, contains all required keys, and has correct values. Ensure `python-dotenv` is installed.
*   **API Key Errors:** Verify the correctness of `LANGSMITH_API_KEY` and the judge LLM's API key in the `.env` file.
*   **`Skipping feedback ... Could not extract prompt/completion` Warnings:** The structure of runs in your LangSmith project might differ from what the `extract_llm_prompt`/`extract_llm_completion` functions expect. Inspect the failing runs in LangSmith UI and adjust the extraction logic in `eval.py` accordingly.
*   **Judge LLM Errors / Parsing Errors:** The judge LLM might not be strictly adhering to the requested JSON output format. Check the logs for parsing errors and potentially adjust the judge prompt or add more robust parsing logic in `eval.py`.

---

*(You can add a similar section for the RAG Evaluation process if needed)*