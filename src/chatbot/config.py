# config.py
import os
from dotenv import load_dotenv

# load_dotenv() is primarily for local development convenience.
# In production, environment variables should be injected by the deployment system (e.g., Kubernetes Secrets).
# It's harmless to leave it here, but it won't load a .env file in a standard container environment.
load_dotenv()

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class Config:
    def __init__(self):
        # --- Database Configuration ---
        # Use os.environ[] which raises KeyError if the variable is missing
        # This ensures the application fails immediately if critical config is absent in production.
        self.POSTGRES_USER = self._get_required_env("POSTGRES_USER")
        self.POSTGRES_PASSWORD = self._get_required_env("POSTGRES_PASSWORD") # Sensitive - MUST be set via secrets
        self.POSTGRES_HOST = self._get_required_env("POSTGRES_HOST")         # e.g., Cloud SQL private IP or proxy hostname
        self.POSTGRES_PORT = self._get_required_int_env("POSTGRES_PORT")
        self.POSTGRES_DB = self._get_required_env("POSTGRES_DB")

        # Construct the database URL dynamically
        self.DATABASE_URL = f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

        # --- Redis Configuration ---
        self.REDIS_HOST = self._get_required_env("REDIS_HOST")               # e.g., Memorystore IP
        self.REDIS_PORT = self._get_required_int_env("REDIS_PORT")
        self.REDIS_DB = self._get_required_int_env("REDIS_DB")

        # --- Application & Logging Settings ---
        # Require LOGGING_LEVEL to be explicitly set (e.g., INFO, DEBUG, WARNING)
        self.LOGGING_LEVEL = self._get_required_env("LOGGING_LEVEL")
        # Useful for distinguishing logs/behaviour between environments
        self.ENVIRONMENT = os.environ.get("ENVIRONMENT", "production") # Default to 'production' if not set

        # --- External API Keys & Secrets ---
        # These MUST be loaded from the environment (via Secrets) and should NOT have defaults
        self.OPENAI_API_KEY = self._get_required_env("OPENAI_API_KEY")
        self.LANGSMITH_API_KEY = self._get_required_env("LANGSMITH_API") # Assuming LANGSMITH_API holds the key
        self.LANGSMITH_PROJECT = self._get_required_env("LANGSMITH_PROJECT")

        # Set LANGSMITH_TRACING based on env var, converting 'true'/'false' string to boolean
        langsmith_tracing_str = os.environ.get("LANGSMITH_TRACING", "false").strip().lower()
        self.LANGSMITH_TRACING = langsmith_tracing_str == 'true'

        self.GROQ_API_KEY = self._get_required_env("GROQ_API")
        self.PINECONE_API_KEY = self._get_required_env("PINECONE_API_KEY") # Ensure this name matches env var
        self.HEVY_API_KEY = self._get_required_env("HEVY_API_KEY")         # Ensure this name matches env var

        # --- ELK/Logging Configuration (Adjust based on your choice in Phase 1, Step 4) ---
        # Option A: Using Cloud Logging (These might not be needed)
        # Option B: Using ELK in Kubernetes (Make these required if using ELK)
        # Using .get() makes them optional if you switch away from ELK later
        self.LOGSTASH_HOST = os.environ.get("LOGSTASH_HOST") # e.g., logstash.monitoring.svc.cluster.local
        logstash_port_str = os.environ.get("LOGSTASH_PORT")
        self.LOGSTASH_PORT = int(logstash_port_str) if logstash_port_str and logstash_port_str.isdigit() else None

    def _get_required_env(self, var_name: str) -> str:
        """Gets a required environment variable or raises ConfigError."""
        value = os.environ.get(var_name)
        if value is None:
            raise ConfigError(f"Missing required environment variable: '{var_name}'")
        return value

    def _get_required_int_env(self, var_name: str) -> int:
        """Gets a required environment variable, converts to int, or raises ConfigError."""
        value_str = self._get_required_env(var_name)
        try:
            return int(value_str)
        except ValueError:
            raise ConfigError(f"Invalid integer value for environment variable '{var_name}': '{value_str}'")

# --- Instantiate the config ---
# This will now raise ConfigError if any required variable defined with _get_required_env is missing
try:
    config = Config()
except ConfigError as e:
    # Log this critical error before raising it further to stop the application
    print(f"CRITICAL CONFIGURATION ERROR: {e}") # Or use a basic logger if available at this stage
    raise SystemExit(1) # Exit the application


# --- How other modules use it (remains the same) ---
# from config import config
#
# db_url = config.DATABASE_URL
# redis_h = config.REDIS_HOST
# level = config.LOGGING_LEVEL
# openai_key = config.OPENAI_API_KEY
# ... etc