export DB_URL="postgresql+psycopg2://$POSTGRES_USER:$POSTGRES_PASSWORD@/$POSTGRES_DB?host=/cloudsql/$INSTANCE_CONNECTION_NAME"


#!/bin/sh
set -e

echo "🔍 Waiting for PostgreSQL to be ready..."

# Ensure password is passed to psql
export PGPASSWORD="$POSTGRES_PASSWORD"

# Try connecting to Postgres for up to 30 seconds
for i in $(seq 1 30); do
  echo "Attempt $i: Connecting to $POSTGRES_HOST:$POSTGRES_PORT as $POSTGRES_USER"
  if psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c '\q' >/dev/null 2>&1; then
    echo "✅ PostgreSQL is available!"
    break
  fi
  sleep 1
done

# If loop exited without success
if [ "$i" = "30" ]; then
  echo "❌ PostgreSQL not available after 30 attempts. Exiting."
  exit 1
fi

echo "🛠 Running Alembic migrations..."
alembic upgrade head

echo "🚀 Starting FastAPI server..."
exec "$@"
