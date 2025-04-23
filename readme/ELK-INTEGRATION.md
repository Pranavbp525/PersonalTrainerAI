# ELK Stack Integration for Fitness Chatbot

This document explains how to integrate the ELK (Elasticsearch, Logstash, Kibana) stack with your fitness chatbot project for comprehensive logging and monitoring.

## Prerequisites

1. Docker and Docker Compose installed
2. Python 3.8+ environment
3. Your fitness chatbot project

## Setup Instructions

### 1. Install Docker Components

Make sure Docker and Docker Compose are installed and running on your system.

### 2. Directory Structure

Ensure you have the following structure in your project:

```
fitness-chatbot/
├── docker-compose.yml
├── elk_logging.py
├── logstash/
│   ├── config/
│   │   └── logstash.yml
│   └── pipeline/
│       └── fitness-chatbot.conf
├── main.py
├── personal_trainer_agent.py
└── ... (other project files)
```

### 3. Launch ELK Stack

You can start the ELK stack with:

```bash
# Make the start script executable
chmod +x start.sh

# Start the ELK stack and application
./start.sh
```

Alternatively, start just the ELK stack with:

```bash
docker-compose up -d
```

### 4. Verify ELK is Running

Verify that all components are running:

- Elasticsearch: http://localhost:9200
- Kibana: http://localhost:5601
- Logstash: Check with `docker-compose logs logstash`

### 5. Integration with Your Python Code

#### Basic Integration

Replace your existing logging setup with the new ELK logger:

```python
from elk_logging import setup_elk_logging

# Instead of:
# logging.basicConfig(...)
# log = logging.getLogger(__name__)

# Use:
log = setup_elk_logging("fitness-chatbot.module_name")
```

#### Agent-Specific Logging

For agent modules, use the agent-specific logger:

```python
from elk_logging import get_agent_logger

async def coordinator(state: AgentState) -> AgentState:
    session_id = state.get("session_id", "unknown")
    log = get_agent_logger("coordinator", session_id)
    
    log.info("Starting coordinator", extra={
        "current_agent": state.get("current_agent")
    })
    
    # Rest of function...
```

#### Performance Tracking

Add timing information to track performance:

```python
import time

async def some_function():
    start_time = time.time()
    
    # Function logic...
    
    duration_ms = int((time.time() - start_time) * 1000)
    log.info("Function completed", extra={
        "duration_ms": duration_ms,
        "other_metrics": value
    })
```

### 6. Kibana Dashboard Setup

Follow the steps in `kibana-dashboard-setup.md` to set up useful dashboards for monitoring your application.

## Log Levels

Use appropriate log levels:

- `log.debug()`: Detailed information, typically for debugging
- `log.info()`: Confirmation that things are working as expected
- `log.warning()`: Indication that something unexpected happened, but the application still works
- `log.error()`: Due to a more serious problem, the application couldn't perform a function
- `log.critical()`: A very serious error that might prevent the application from continuing to run

## Structured Logging

Use the `extra` parameter to add structured data to your logs:

```python
log.info("User logged in", extra={
    "user_id": user_id,
    "login_method": "password"
})
```

## LLMOPS-Specific Logging

For LLM operations, include these fields:

```python
log.info("LLM invocation completed", extra={
    "model": "gpt-4",
    "prompt_tokens": 250,
    "completion_tokens": 100,
    "duration_ms": 3200,
    "agent": "planning_agent"
})
```

## Troubleshooting

### Logstash Connection Issues

If your application can't connect to Logstash:

1. Check that the ELK stack is running: `docker-compose ps`
2. Verify the Logstash port is open: `nc -zv localhost 5000`
3. Check Logstash logs: `docker-compose logs logstash`

### Missing Logs in Kibana

If logs aren't appearing in Kibana:

1. Check your index pattern in Kibana
2. Verify logs are being sent to Logstash: `docker-compose logs logstash | grep received`
3. Verify Elasticsearch is receiving data: `curl -X GET "localhost:9200/_cat/indices?v"`

## Production Deployment

For production:

1. Configure security (X-Pack)
2. Set up persistent storage
3. Scale Elasticsearch to multiple nodes
4. Configure backup/restore
5. Set up alerting for production monitoring

Consult the ELK documentation for more details on production deployment.