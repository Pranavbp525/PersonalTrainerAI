# redis_utils.py
import redis
import logging
from .config import config  # Import the config

log = logging.getLogger(__name__)

def get_redis_client():
    """
    Gets a Redis client with connection pooling.
    """
    try:
        pool = redis.ConnectionPool(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            decode_responses=True  # Decode strings automatically
        )
        client = redis.Redis(connection_pool=pool)
        client.ping()  # Test the connection
        return client
    except redis.exceptions.RedisError as e:
        log.error(f"Error connecting to Redis: {e}")
        return None  # Or raise the exception, depending on your needs


def store_chat_history(session_id: str, messages: list):
    """Stores chat history in Redis."""
    client = get_redis_client()
    if client:
        try:
            key = f"session:{session_id}:history"
            # Use a transaction (pipeline) for efficiency
            with client.pipeline() as pipe:
                pipe.delete(key)  # Clear existing history (optional)
                for message in messages:
                    pipe.rpush(key, message)  # Or serialize to JSON
                pipe.execute()
        except redis.exceptions.RedisError as e:
            log.error(f"Error storing chat history in Redis: {e}")

def get_chat_history(session_id: str) -> list:
    """Retrieves chat history from Redis."""
    client = get_redis_client()
    if client:
        try:
            key = f"session:{session_id}:history"
            history = client.lrange(key, 0, -1)
            return history
        except redis.exceptions.RedisError as e:
            log.error(f"Error getting chat history from Redis: {e}")
            return []  # Return empty list on error
    return []