# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY src/ /app/src/

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Copy entrypoint script
COPY run_migrations.sh /app/

# Set execute permissions on entrypoint script
RUN chmod +x /app/run_migrations.sh

# Define entrypoint
ENTRYPOINT ["/app/run_migrations.sh"]

# Run uvicorn when the container launches
CMD ["uvicorn", "src.chatbot.main:app", "--host", "0.0.0.0", "--port", "8000"]