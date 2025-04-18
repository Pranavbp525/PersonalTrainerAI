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
(NOTE YOU NEED YOUR LOCAL ALEMBIC STRUCTURE TO BUILD CONTEXT FOR DOCKER IMAGE!)
1. Repeat the step of creating alembic schema if you haven't already and configure alembic.ini and alembic/env.py
### BEFORE PROCEEDING MAKE SURE YOU HAVE YOUR VERSIONS IN alembic/versions ---
2. Configure .env.local in root.
3. Bring the stack up in the background
```
docker compose -f docker-compose.merged.yml up -d
```
4. Generate a brand‑new “initial schema” migration inside the API container
```
docker compose -f docker-compose.merged.yml exec api `
  /bin/sh -c "cd src/chatbot && alembic revision --autogenerate -m 'initial schema'"
```
5. Apply the migration
```
docker compose -f docker-compose.merged.yml exec api `
  /bin/sh -c "cd src/chatbot && alembic upgrade head"
```
6. Restart the API service
```
docker compose -f docker-compose.merged.yml restart api
```
7. Verify the tables exist
```
docker compose -f docker-compose.merged.yml exec postgres `
  psql -U postgres -d chatbot_db -c "\dt"
```
