#!/bin/bash

# Set environment variables
export LOGSTASH_HOST=localhost
export LOGSTASH_PORT=5044  # Changed from 5000
export ENVIRONMENT=development

# Check if docker-compose or docker compose is available
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker compose"
else
    echo "Neither docker-compose nor docker compose is available. Please install Docker Compose."
    exit 1
fi

echo "Using Docker Compose command: $DOCKER_COMPOSE_CMD"

# Start ELK stack with specific compose file
echo "Starting ELK stack..."
$DOCKER_COMPOSE_CMD -f docker-compose-elk.yml up -d

# Wait for Elasticsearch to be ready
echo "Waiting for Elasticsearch to be ready..."
until $(curl --output /dev/null --silent --head --fail http://localhost:9200); do
    printf '.'
    sleep 5
done
echo "Elasticsearch is ready!"

# Wait for Kibana to be ready
echo "Waiting for Kibana to be ready..."
until $(curl --output /dev/null --silent --head --fail http://localhost:5601); do
    printf '.'
    sleep 5
done
echo "Kibana is ready!"

# Optionally start your application (comment this if you use another script for your main app)
# echo "Starting fitness chatbot application..."
# cd src/chatbot && uvicorn main:app --reload