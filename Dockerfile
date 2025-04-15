# ---- Stage 1: Build Stage ----
# Use a standard Python image with build tools
FROM python:3.9 AS builder

# Set environment variables (Correct Format: key=value)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the builder stage
WORKDIR /app

# Create a virtual environment
RUN python -m venv /opt/venv
# Make venv available to subsequent RUN commands in this stage
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install OS dependencies needed for building Python packages if necessary
# Example: RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev libpq-dev && rm -rf /var/lib/apt/lists/*
# Consider using psycopg[binary] in requirements.txt to potentially avoid needing these build tools here.

# Install Python dependencies into the virtual environment
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy the rest of the application source code and necessary files
COPY ./src ./src
COPY ./alembic ./alembic
COPY alembic.ini .

# ---- Stage 2: Final Production Stage ----
# Use a slim Python image for a smaller footprint
FROM python:3.9-slim

# Set environment variables (Correct Format: key=value)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install OS dependencies needed at RUNTIME
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq-dev gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory for the final stage
WORKDIR /app

# Create a non-root user and group
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Copy the virtual environment from the builder stage
COPY --from=builder --chown=appuser:appgroup /opt/venv /opt/venv

# Copy the application code and other necessary files from the builder stage
COPY --from=builder --chown=appuser:appgroup /app/src ./src
COPY --from=builder --chown=appuser:appgroup /app/alembic ./alembic
COPY --from=builder --chown=appuser:appgroup /app/alembic.ini .

# Create and set permissions for Hugging Face cache
RUN mkdir /app/.cache && chown appuser:appgroup /app/.cache

# Set environment variable for Hugging Face cache location
ENV HF_HOME=/app/.cache

# Set the PATH environment variable to include the venv's bin directory
ENV PATH="/opt/venv/bin:$PATH"

# Switch to the non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the application
CMD exec /bin/sh -c "cd /app/src/chatbot && exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2"