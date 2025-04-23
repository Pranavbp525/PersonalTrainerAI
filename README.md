
# PersonalTrainerAI

## Overview

PersonalTrainerAI is an AI-powered personal fitness trainer that provides customized workout plans, real-time feedback, and progress tracking. Using advanced AI Agent systems, it adapts to individual fitness levels and goals, ensuring a personalized training experience.

### Key Features

- **Personalized Workout Planning**: AI-generated routines based on user profile and goals
- **Progress Tracking & Adaptation**: Analyzes workout logs to track progress and suggest routine adjustments
- **RAG-Powered Knowledge**: Leverages fitness science through Retrieval Augmented Generation
- **Multi-Agent Architecture**: Specialized agents for research, planning, analysis, and coaching
- **Hevy Integration**: Direct creation and modification of routines in the Hevy app

## Quick Start

### Prerequisites
- Python 3.8+
- Docker & Docker Compose
- Google Cloud SDK (for deployment)

### Installation
```bash
git clone https://github.com/yourusername/PersonalTrainerAI.git
cd PersonalTrainerAI
pip install -r requirements.txt
```

### Local Development

**1. Database Setup:** 
- Install PostgreSQL if you don't already have it
- Create a database called `chatbot_db`
- Locate the `schema.sql` file in the project root directory
- Run the schema file using either:
  ```bash
  psql -U your_username -d chatbot_db -a -f schema.sql
  ```
  Or from within psql:
  ```sql
  \i schema.sql
  ```

**2. Setting up env variables:** Fill in all the values inside .env.local file.

**3. Run Docker Compose:**
```bash
# Start the backend and frontend services
docker-compose -f docker-compose.local.yml up
```

Visit `http://localhost:8501` to access the frontend.

## Documentation

For more detailed documentation on specific components:

- [**AI Agent Architecture**](./readme/agent.md): Multi-agent system design and components
- [**Data Pipeline**](./readme/data_pipeline.md): Data collection, processing, and storage
- [**RAG Implementation**](./readme/MLflow.md): Retrieval-augmented generation models
- [**Deployment Guide**](./readme/deployment.md): Cloud deployment and CI/CD pipeline
- [**Model Development**](./readme/model_development.md): Agent development and evaluation
- [**Airflow Integration**](./readme/airflow.md): Workflow orchestration
- [**User Guide**](./readme/user_guide.md): End-user instructions
- [**Evaluation Framework**](./readme/evaluation.md): Quality assessment and performance metrics
- [**Maintenance Guide**](./readme/maintenance.md): System upkeep and troubleshooting
- [**System Architecture**](./readme/architecture.md): High-level technical design overview


## Project Structure

The repository is organized as follows:

- `src/chatbot/`: Core chatbot application (FastAPI backend + Streamlit frontend)
- `src/chatbot/agent/`: AI agent system implementation using LangGraph
- `src/data_pipeline/`: Data collection and preprocessing scripts
- `src/rag_model/`: RAG implementations and evaluation
- `dags/`: Airflow DAG definitions for automation
- `tests/`: Unit and integration tests
- `.github/workflows/`: CI/CD pipeline configurations

## Contributing

Please read our [Contributing Guidelines](./CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the [MIT License](LICENSE).
```
