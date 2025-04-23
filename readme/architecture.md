# PersonalTrainerAI Architecture Overview

This document provides a high-level overview of the PersonalTrainerAI system architecture.

## System Components

```
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                   │     │                   │     │                   │
│   Client Layer    │     │   Service Layer   │     │    Data Layer     │
│                   │     │                   │     │                   │
└────────┬──────────┘     └────────┬──────────┘     └────────┬──────────┘
         │                         │                         │
         ▼                         ▼                         ▼
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│  Streamlit UI     │     │  FastAPI Backend  │     │  Vector Database  │
│  (chat_client.py) │────▶│  (main.py)        │────▶│  (Pinecone)       │
└───────────────────┘     └────────┬──────────┘     └───────────────────┘
                                   │                          ▲
                                   ▼                          │
                          ┌───────────────────┐               │
                          │  Agent System     │               │
                          │  (LangGraph)      │───────────────┘
                          └────────┬──────────┘
                                   │
                                   ▼
                          ┌───────────────────┐     ┌───────────────────┐
                          │  External APIs    │     │  Database         │
                          │  - Hevy           │────▶│  - PostgreSQL     │
                          │  - OpenAI/Claude  │     │  - Redis          │
                          └───────────────────┘     └───────────────────┘
```

## Key Components

### Client Layer
- **Streamlit Frontend**: User interface for interacting with the AI trainer
- Handles user authentication, chat history, and message rendering

### Service Layer
- **FastAPI Backend**: Core API service handling business logic
- **LangGraph Agent System**: Multi-agent framework for specialized fitness tasks:
  - Coordinator Agent: Central router managing conversation flow
  - User Modeler: Builds and updates user profile
  - Research Agent: Retrieves scientific fitness information
  - Planning Agent: Creates workout routines
  - Progress Analysis Agent: Reviews workout history and suggests adaptations
  - Coach Agent: Provides motivation and adherence strategies

### Data Layer
- **Pinecone Vector DB**: Stores embeddings of fitness knowledge for RAG
- **PostgreSQL**: Persistent storage for user profiles, sessions, messages
- **Redis**: Cache for chat history and temporary storage

### External Integration
- **Hevy API**: Integration with Hevy workout app for routine management
- **LLM APIs**: Integration with OpenAI or Claude for agent intelligence

## Data Flow

1. User submits message through Streamlit UI
2. Backend receives message, creates entry in PostgreSQL
3. Agent system processes message:
   - Coordinator routes to appropriate specialized agent
   - Agent may query vector DB for fitness knowledge
   - Agent may use external APIs (e.g., Hevy)
4. Response generated and stored in PostgreSQL
5. Frontend retrieves and displays response

## Deployment Architecture

The system is deployed on Google Cloud Platform using:
- Cloud Run for containerized services
- Cloud SQL for PostgreSQL database
- Artifact Registry for container images
- Secret Manager for API keys and credentials
- Cloud Storage for evaluation artifacts
```