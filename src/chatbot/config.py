# config.py
import os
from dotenv import load_dotenv



load_dotenv()  # Load environment variables from .env file (if it exists)

LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL", "DEBUG")

class Config:
    # Database Configuration
    POSTGRES_USER = os.environ.get("POSTGRES_USER", "defaultuser")
    POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "defaultpassword")
    POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", 5432))
    POSTGRES_DB = os.environ.get("POSTGRES_DB", "chatbot_db")
    DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    

    # Redis Configuration
    REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
    REDIS_DB = int(os.environ.get("REDIS_DB", 0))

    # Other settings (e.g., API keys, logging level)
    LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL", "INFO")

config = Config()