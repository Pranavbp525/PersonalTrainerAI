# redis_utils.py
import redis
import logging
# --- Imports the centralized config ---
from config import config

# Use a logger specific to this module
log = logging.getLogger(__name__) # Standard Python logging

# Keep a global pool for efficiency
_redis_pool = None

def get_redis_pool():
    """Initializes and returns the Redis connection pool."""
    global _redis_pool
    if _redis_pool is None:
        try:
            # --- Use config object for connection details ---
            _redis_pool = redis.ConnectionPool(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                db=config.REDIS_DB,
                decode_responses=True, # Decode strings automatically
                socket_timeout=5,      # Add a timeout
                socket_connect_timeout=5 # Add a connection timeout
            )
            # Test connection during pool creation
            temp_client = redis.Redis(connection_pool=_redis_pool)
            temp_client.ping()
            log.info(f"Redis connection pool created and connected to {config.REDIS_HOST}:{config.REDIS_PORT}")
        except redis.exceptions.ConnectionError as e:
            log.error(f"Error connecting to Redis ({config.REDIS_HOST}:{config.REDIS_PORT}): {e}")
            _redis_pool = None # Ensure pool remains None on error
            # Optionally re-raise or handle differently depending on desired app behavior
            raise  # Re-raise to prevent app from potentially starting without Redis if critical
        except Exception as e:
            log.error(f"Unexpected error creating Redis pool ({config.REDIS_HOST}:{config.REDIS_PORT}): {e}")
            _redis_pool = None
            raise
    return _redis_pool


def get_redis_client():
    """Gets a Redis client from the connection pool."""
    pool = get_redis_pool()
    if pool:
        return redis.Redis(connection_pool=pool)
    else:
        # This case should ideally not be reached if get_redis_pool() raises on error
        log.error("Redis connection pool is not available.")
        return None

def store_chat_history(session_id: str, messages: list):
    """Stores chat history in Redis."""
    client = get_redis_client()
    if client:
        try:
            key = f"session:{session_id}:history"
            # Use a transaction (pipeline) for efficiency
            with client.pipeline() as pipe:
                pipe.delete(key)
                if messages: # Only push if there are messages
                    pipe.rpush(key, *messages) # Unpack list items for rpush
                # Set an expiration time (e.g., 7 days) - ADJUST AS NEEDED
                pipe.expire(key, 60 * 60 * 24 * 7)
                pipe.execute()
            log.debug(f"Stored/updated chat history in Redis for session: {session_id}")
        except redis.exceptions.RedisError as e:
            log.error(f"Redis error storing chat history for session {session_id}: {e}")
        except Exception as e:
            log.error(f"Unexpected error storing chat history for session {session_id}: {e}")


def get_chat_history(session_id: str) -> list:
    """Retrieves chat history from Redis."""
    client = get_redis_client()
    if client:
        try:
            key = f"session:{session_id}:history"
            history = client.lrange(key, 0, -1)
            log.debug(f"Retrieved {len(history)} messages from Redis history for session: {session_id}")
            return history
        except redis.exceptions.RedisError as e:
            log.error(f"Redis error getting chat history for session {session_id}: {e}")
            return []
        except Exception as e:
            log.error(f"Unexpected error getting chat history for session {session_id}: {e}")
            return []
    # This part is less likely if get_redis_client() is robust, but keep for safety
    log.warning(f"Redis client not available. Could not retrieve history for session: {session_id}")
    return []