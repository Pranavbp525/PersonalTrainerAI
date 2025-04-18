# Chatbot Application Setup and Usage Guide

## Prerequisites
1. Install PostgreSQL and Redis locally
2. Create a `.env` file with required configurations (refer to `config.py` for required fields)

## Setup Instructions
1. Initialize Alembic:
   ```bash
   alembic init alembic
   ```
2. Perform database migration:
   ```bash
   alembic revision --autogenerate -m "initial migration"
   alembic upgrade head
   ```

## Verification
1. Check if databases are created in your system:
   - Verify PostgreSQL database exists
   - Verify Redis is accessible

## Running the Application
1. Start Redis server:
   ```bash
   redis-server
   ```
2. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```
3. Start the chat client:
   ```bash
   python chat_client.py
   ```

## Usage
1. When prompted, enter a unique username
2. A new user will be created in the database with a session
3. Interact with the chatbot:
   - General queries: Direct answers
   - Workout queries: Retrieves from RAG and creates workout routines
4. All chat history is permanently stored in PostgreSQL

## Notes
- Ensure Redis and PostgreSQL services are running before starting the application
- Check logs for any connection issues
- Session management and chat history are handled automatically

# RUNNING DOCKER

> **Prerequisite:** Make sure your repo includes  
> - `alembic.ini`  
> - the full `alembic/versions/` folder  
> - `env.py` configured to read your `DATABASE_URL`

1. Configure .env.local in root.
In .env.local, ensure POSTGRES_HOST=postgres, REDIS_HOST=redis, etc.

2. Bring the stack up in the background
```
docker compose -f docker-compose.merged.yml up --build -d
```
3. Apply existing migrations
```
docker compose -f docker-compose.merged.yml exec api `
  /bin/sh -c "cd src/chatbot && alembic upgrade head"
```
4. Restart the API
```
docker compose -f docker-compose.merged.yml restart api
```

5. Confirm table exists
```
docker compose -f docker-compose.merged.yml exec postgres \
    psql -U postgres -d chatbot_db -c "\dt"
```

