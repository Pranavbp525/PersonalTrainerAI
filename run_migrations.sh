#!/bin/bash

# Wait for PostgreSQL to be ready and the database to exist
until psql -h db -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c '\q' &> /dev/null; do
  echo "Waiting for PostgreSQL and the database to become available..."
  sleep 1
done

# Set the SQLALCHEMY_URL environment variable for Alembic
export SQLALCHEMY_URL="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@db:${POSTGRES_PORT}/${POSTGRES_DB}"

# Run Alembic migrations
echo "Running Alembic migrations..."
if ! alembic upgrade head; then
  echo "Alembic migrations failed!"
  exit 1  # Exit with an error code if migrations fail
fi

# Start the application (this will be overridden by CMD in the Dockerfile)
echo "Migrations complete.  Starting application..."
exec "$@"