# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Install PostgreSQL client
RUN apt-get update && apt-get install -y postgresql-client

# Copy the current directory contents into the container at /app
COPY src/ /app/src/

# Copy entrypoint script
COPY run_migrations.sh /app/

# Set execute permissions on entrypoint script
RUN chmod +x /app/run_migrations.sh

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define entrypoint
ENTRYPOINT ["/app/run_migrations.sh"]

# Run uvicorn when the container launches
CMD ["uvicorn", "src.chatbot.main:app", "--host", "0.0.0.0", "--port", "8000"]
