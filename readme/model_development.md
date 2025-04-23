# Model Development - AI Personal Fitness Trainer

## 1. Overview

This document outlines the model development process for the AI Personal Fitness Trainer project. Unlike traditional ML model development involving training on large datasets, this project focuses on designing, implementing, and evaluating a sophisticated AI Agent built using **LangGraph**.

The core "model" is a multi-agent system orchestrated by LangGraph. This agent interacts with users, tracks workout logs, accesses fitness knowledge via a Retrieval-Augmented Generation (RAG) system, adapts routines based on progress, and creates new routines based on user profiles.

Key technologies used in model development include:

*   **LangGraph:** For defining the agent's state machine and multi-agent workflows.
*   **Langchain:** For LLM interactions, prompt management, and tool usage.
*   **Pinecone:** As the vector store backing the RAG system for fitness knowledge.
*   **PostgreSQL:** For persistent agent state storage via `AsyncPostgresSaver` (LangGraph Checkpointer) and also for chat, users and sessions storage.
*   **Langsmith:** For comprehensive tracing, observability, prompt versioning, and agent evaluation.
*   **Google Cloud Platform (GCP) Buckets:** For storing evaluation artifacts.
*   **ELK Stack (Elasticsearch, Logstash, Kibana):** For structured application and agent logging.
*   **FastAPI:** For serving the compiled LangGraph agent as an interactive chatbot API.

## 2. Model Architecture: The LangGraph Agent

The heart of the system is a multi-agent graph defined in `graph.py`. This graph manages the conversation flow and delegates tasks to specialized agents or tools.

**Core Components:**

1.  **State Management:** Uses `AgentState` (and specialized state Pydantic models like `DeepFitnessResearchState`) to manage information throughout the conversation. State is persisted using `AsyncPostgresSaver` connected to a PostgreSQL database, allowing users to resume conversations.
2.  **Coordinator Agent:** The central router (`coordinator` node) analyzes the user's request and the current state to determine the next step, routing to the appropriate specialized agent or tool. See `COORDINATOR_TEMPLATE` for its logic.
3.  **Specialized Agents (Nodes):**
    *   **User Modeler (`user_modeler`):** Updates the user's profile based on interactions. (See `USER_MODELER_TEMPLATE`)
    *   **Deep Research (`deep_research` subgraph):** Performs in-depth research using the RAG system to answer complex fitness questions or gather information for planning. It involves planning steps, generating queries, executing RAG, synthesizing results, and reflecting. (See `RESEARCH_TEMPLATE`, `TARGETED_RAG_QUERY_TEMPLATE`)
    *   **Streamlined Routine Creation (`planning_agent` subgraph):** Generates new, personalized workout routines formatted for the Hevy app, based on user profile and research findings. (See `PLANNING_TEMPLATE`)
    *   **Progress Analysis & Adaptation (`progress_adaptation_agent` subgraph):** Analyzes user workout logs (`tool_fetch_workouts`), identifies routines needing adaptation (`ROUTINE_IDENTIFICATION_PROMPT`), performs analysis (`ANALYSIS_TEMPLATE_V2`), potentially uses RAG for adaptation strategies (`TARGETED_RAG_QUERY_TEMPLATE`), modifies routines (`ROUTINE_MODIFICATION_TEMPLATE_V2`, `tool_update_routine`), and generates reasoning/reports (`REASONING_GENERATION_TEMPLATE`, `FINAL_CYCLE_REPORT_TEMPLATE_V2`).
    *   **Coach Agent (`coach_agent`):** Provides motivation, adherence tips, and encouragement. (See `COACH_TEMPLATE`)
    *   **End Conversation (`end_conversation`):** Handles the end of an interaction flow.
4.  **Tools Node (`tools`):** Executes functions like fetching/updating data from Hevy (`tool_fetch_workouts`, `tool_update_routine`, etc.) or querying the RAG system (`retrieve_from_rag`).
5.  **Routing Logic:** Conditional edges (`coordinator_condition`, `check_completion_and_route_v2`, `_check_targets_found`) direct the flow based on the state and agent outputs.
6.  **Error Handling:** Agents are wrapped (`agent_with_error_handling`) to catch exceptions, log them (via ELK), and attempt graceful recovery by routing back to the coordinator.
7.  **Logging:** Structured logging using a custom ELK setup (`elk_logging.py`) provides detailed insights into agent execution alongside Langsmith traces.

## 3. Model Development and Evaluation Pipeline

The development process follows MLOps best practices adapted for LLM-based agent systems.

1.  **Loading Data:**
    *   **RAG Data Source:** The primary external knowledge source is the fitness information stored in the **Pinecone** vector index. This data is loaded on-demand via the `retrieve_from_rag` tool when agents like "Deep Research" or "Progress Analysis & Adaptation" require it. The data pipeline responsible for populating Pinecone is considered upstream.
    *   **Agent State:** The conversational state, including user profile, messages, and working memory, is loaded from the **PostgreSQL checkpointer** at the start of each interaction for a given session ID.

2.  **"Training" and Selecting the Best "Model":**
    *   This phase involves **Agent Design and Prompt Engineering** rather than traditional model training.
    *   The "model" is the collection of agent logic (`graph.py`), prompt templates (defined in Python constants), and configurations.
    *   **Prompt Versioning:** Prompts are managed and versioned within the codebase and tracked via **Langsmith**, allowing experimentation and rollback.
    *   Selection of the "best model" involves iterative refinement of agent logic, routing, tool usage, and prompts based on evaluation results and qualitative analysis of traces in Langsmith.

3.  **Model Validation & Evaluation:**
    *   Validation is performed through a **continuous evaluation pipeline**.
    *   **Process:** A cron job runs every 2 weeks, executing predefined test scenarios or datasets against the agent.
    *   **Evaluation Method:** **Langsmith** is used extensively. It employs an "LLM-as-judge" approach where another LLM evaluates the agent's traces based on predefined criteria (e.g., correctness, helpfulness, adherence to instructions, safety).
    *   **Metrics:** Each run in Langsmith receives an evaluation score. Key metrics tracked include this score, latency, cost, and token usage.
    *   **Artifacts:** Detailed evaluation results and run traces are stored in **Langsmith**. Aggregated reports or summaries from the cron job evaluation are stored as artifacts in a **GCP Bucket**.
    *   **Drift Detection:** If the average evaluation score from the cron job drops below a predefined threshold, it indicates potential performance degradation or drift. An **email notification** is triggered to alert the development team.

4.  **Model Bias Detection (Using Evaluation & Analysis):**
    *   **Primary Method:** Bias is primarily monitored through the **Langsmith evaluation pipeline**. While not explicitly using tools like Fairlearn or TFMA slicing in the traditional sense, the evaluation dataset *can* be designed to include diverse user profiles or scenarios. Analyzing evaluation scores across these different slices helps identify potential performance disparities.
    *   **Trace Analysis:** Manually reviewing traces in Langsmith for interactions with potentially sensitive attributes or demographic groups helps identify subtle biases in language or recommendations.
    *   **Prompt Design:** Prompts are designed to be objective and avoid incorporating harmful stereotypes. They explicitly guide the agent towards evidence-based fitness advice.
    *   **Mitigation:** If bias is detected (e.g., consistently lower scores for certain scenarios, biased language in traces), mitigation involves refining prompts, adjusting agent logic, or potentially adding explicit fairness constraints or checks within the agent's reasoning steps.

5.  **Pushing the Model to Registry/Deployment:**
    *   The "model" in this context is the entire application codebase, including the LangGraph definition (`graph.py`), agent prompts, dependencies, and the FastAPI wrapper.
    *   **Containerization:** The application is containerized using Docker (assumed, standard practice).
    *   **Registry:** The Docker image containing the agent application is pushed to a container registry (e.g., Google Artifact Registry, Docker Hub).
    *   **Deployment:** The container is deployed to the serving environment. Updates involve building a new image with code/prompt changes and redeploying. Langsmith prompt versions provide a registry for the prompt component of the model.

## 4. Hyperparameter Tuning

Traditional hyperparameter tuning (e.g., learning rate) is not applicable. "Tuning" in this context involves:

*   **LLM Selection:** Choosing the underlying LLM(s) for the agents (e.g., GPT-4, Claude 3).
*   **LLM Parameters:** Adjusting parameters like temperature or top_p for creativity vs. factuality.
*   **Prompt Engineering:** Iteratively refining prompts for clarity, effectiveness, and safety (tracked via Langsmith).
*   **Agent Configuration:** Modifying agent-specific settings (e.g., `max_iterations` in the research loop).
*   **Tool Configuration:** Adjusting how tools are used or their parameters.

These are tuned based on evaluation results and trace analysis in Langsmith.

## 5. Experiment Tracking and Results (Langsmith)

**Langsmith** is the central tool for experiment tracking:

*   **Automated Tracing:** Captures every LLM call, tool execution, agent step, input/output, latency, and cost automatically.
*   **Prompt Playground & Versioning:** Facilitates experimenting with prompts and tracks versions.
*   **Evaluation Runs:** Stores detailed results from the automated evaluation pipeline (LLM-as-judge scores).
*   **Datasets:** Allows curation of datasets for testing and evaluation.
*   **Monitoring:** Provides dashboards to track performance metrics over time.

Results (evaluation scores, pass/fail rates, qualitative examples) are analyzed directly within Langsmith and supplemented by artifacts stored in GCP Buckets.

## 6. Model Sensitivity Analysis

Sensitivity is analyzed by observing how the agent's behavior changes in response to:

*   **Input Variations:** Testing with diverse user queries, profiles, and edge cases.
*   **Prompt Changes:** Comparing traces before and after prompt modifications (leveraging Langsmith's comparison views).
*   **RAG Results:** Assessing how different retrieved document chunks affect agent reasoning and output.
*   **LLM Changes:** Evaluating performance if the underlying LLM version is updated.
*   **Tool Failures:** Observing agent recovery behavior when tools return errors.

Langsmith traces are crucial for this analysis.

## 7. CI/CD Pipeline Automation for Model Development

The primary CI/CD mechanism focused on *model quality* is the **Automated Evaluation Pipeline**:

1.  **Trigger:** Runs on a schedule (every 2 weeks via cron job).
2.  **Steps:**
    *   Fetches the latest deployed agent version.
    *   Runs a predefined evaluation dataset against the agent.
    *   Uses Langsmith LLM-as-judge to score each interaction trace.
    *   Aggregates scores and stores results/artifacts (Langsmith, GCP Bucket).
    *   Checks if the average score meets the required threshold.
    *   Sends email notifications on drift detection (score below threshold) or pipeline failures.
3.  **Purpose:** Ensures ongoing model quality, detects regressions or drift, and provides continuous feedback on the agent's performance.

*(Note: A separate CI/CD pipeline would handle the build, test, and deployment of the application code/container itself whenever changes are pushed to the repository).*

## 8. Code Implementation Details

*   **Core Logic:** `graph.py` contains the LangGraph `StateGraph` definition, including nodes (agents, tools, subgraphs), edges, and state models.
*   **Agent Prompts:** Defined as constants within the codebase (e.g., `COORDINATOR_TEMPLATE`, `PLANNING_TEMPLATE`).
*   **State:** Defined using Pydantic models (`AgentState`, etc.) for type safety and clarity. Persistence handled by `AsyncPostgresSaver`.
*   **Tools:** Defined in `agent/llm_tools.py` using Langchain tool decorators/classes.
*   **Logging:** Integrated ELK logging (`elk_logging.py`) for application-level and agent-specific logs.
*   **Serving:** The compiled LangGraph `app` is exposed via **FastAPI** endpoints for user interaction.

This comprehensive approach ensures the AI Personal Fitness Trainer is robust, observable, evaluable, and maintainable.