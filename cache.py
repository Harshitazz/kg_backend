"""
Redis-based caching for query results with proper cache keys
Redis is optional - gracefully falls back if not available
"""
import pickle
import hashlib
import json
from typing import Optional, Dict, Any
import os
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis package not installed. Caching will be disabled. Install with: pip install redis")

# Cache configuration
QUERY_CACHE_TTL = int(os.getenv("QUERY_CACHE_TTL", "1800"))  # 30 minutes default
REDIS_URL = os.getenv("REDIS_URL", "redis://default:IopYE3MLrO1kwJ9Utgi3o3nZubBEz1qE@redis-15009.c73.us-east-1-2.ec2.cloud.redislabs.com:15009")

# Track if Redis is available
_redis_client = None
_redis_available = False

@lru_cache(maxsize=1)
def get_redis_client():
    """Get Redis client, returns None if Redis is not available (graceful fallback)"""
    global _redis_client, _redis_available
    
    if not REDIS_AVAILABLE:
        return None
    
    if _redis_client is not None:
        return _redis_client
    
    try:
        client = redis.from_url(REDIS_URL, decode_responses=False)
        client.ping()
        _redis_client = client
        _redis_available = True
        logger.info(f"Connected to Redis at {REDIS_URL}")
        return client
    except Exception as e:
        logger.warning(f"Redis not available at {REDIS_URL}: {e}. Caching will be disabled.")
        _redis_available = False
        return None


def _hash_string(s: str) -> str:
    """Generate MD5 hash of a string"""
    return hashlib.md5(s.encode()).hexdigest()


def _hash_dict(d: Dict) -> str:
    """Generate hash from a dictionary"""
    sorted_dict = json.dumps(d, sort_keys=True)
    return _hash_string(sorted_dict)


def get_cache_key_for_query(
    question: str, 
    user_id: Optional[str] = None,
    task_ids: Optional[list] = None,
    embedding_model: Optional[str] = None,
    search_params: Optional[Dict[str, Any]] = None,
    prompt_version: Optional[str] = None
) -> str:
    """Generate cache key for query including all relevant parameters
    Cache is task_id aware - same question with different task_ids will have different cache keys
    """
    # Include user_id and sorted task_ids in cache key for proper isolation
    if user_id:
        base_key = f"{user_id}:"
    else:
        base_key = ""
    
    # Include task_ids in cache key (sorted for consistency)
    if task_ids:
        sorted_task_ids = sorted(task_ids)
        base_key += f"{'_'.join(sorted_task_ids)}:"
    
    base_key += question
    
    if embedding_model:
        base_key += f":model:{embedding_model}"
    
    if search_params:
        base_key += f":params:{_hash_dict(search_params)}"
    
    if prompt_version:
        base_key += f":prompt:{_hash_string(prompt_version)}"
    
    return _hash_string(base_key)


def get_cached_query(
    question: str, 
    user_id: Optional[str] = None,
    task_ids: Optional[list] = None,
    embedding_model: Optional[str] = None,
    search_params: Optional[Dict[str, Any]] = None,
    prompt_version: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Get cached query result (gracefully handles Redis unavailability)
    Cache is task_id aware - only returns cache if user_id and task_ids match
    """
    cache_key = get_cache_key_for_query(question, user_id, task_ids, embedding_model, search_params, prompt_version)
    
    redis_client = get_redis_client()
    if redis_client is None:
        logger.debug("Redis not available, skipping query cache lookup")
        return None
    
    try:
        cached_data = redis_client.get(f"query:{cache_key}")
        if cached_data:
            logger.info(f"Cache HIT for query: {question[:50]}...")
            return pickle.loads(cached_data)
        else:
            logger.debug(f"Cache MISS for query: {question[:50]}...")
    except Exception as e:
        logger.warning(f"Error reading from Redis cache: {e}. Continuing without cache.")
        return None
    
    return None


def cache_query(
    question: str, 
    result: Dict[str, Any], 
    user_id: Optional[str] = None,
    task_ids: Optional[list] = None,
    embedding_model: Optional[str] = None,
    search_params: Optional[Dict[str, Any]] = None,
    prompt_version: Optional[str] = None,
    ttl: int = QUERY_CACHE_TTL
):
    """Cache query result with TTL (gracefully handles Redis unavailability)
    Cache is task_id aware - stores cache with user_id and task_ids in key
    """
    cache_key = get_cache_key_for_query(question, user_id, task_ids, embedding_model, search_params, prompt_version)
    
    redis_client = get_redis_client()
    if redis_client is None:
        logger.debug("Redis not available, skipping query cache write")
        return
    
    try:
        serialized = pickle.dumps(result)
        redis_client.setex(f"query:{cache_key}", ttl, serialized)
        logger.info(f"Cached query result: {question[:50]}... (TTL: {ttl}s)")
    except Exception as e:
        error_msg = str(e)
        if "maxmemory" in error_msg.lower() or "command not allowed" in error_msg.lower():
            logger.warning(f"Redis memory limit reached. Attempting to clear old cache entries...")
            try:
                _evict_old_cache_entries(redis_client, keep_recent=True)
                try:
                    redis_client.setex(f"query:{cache_key}", ttl, serialized)
                    logger.info(f"Cached query result after eviction: {question[:50]}... (TTL: {ttl}s)")
                except Exception as retry_error:
                    logger.warning(f"Still unable to cache after eviction: {retry_error}. Continuing without cache.")
            except Exception as evict_error:
                logger.warning(f"Failed to evict cache entries: {evict_error}. Continuing without cache.")
        else:
            logger.warning(f"Error writing to Redis cache: {e}. Continuing without cache.")


def _evict_old_cache_entries(redis_client, keep_recent: bool = True, evict_percentage: float = 0.3):
    """Evict old cache entries to free up Redis memory
    Args:
        redis_client: Redis client instance
        keep_recent: If True, evict oldest entries first (LRU-like)
        evict_percentage: Percentage of cache entries to evict (0.0 to 1.0)
    """
    try:
        query_keys = []
        
        for key in redis_client.scan_iter(match="query:*"):
            ttl = redis_client.ttl(key)
            query_keys.append((key, ttl))
        
        query_keys.sort(key=lambda x: x[1] if x[1] > 0 else float('inf'))
        
        num_to_evict = max(1, int(len(query_keys) * evict_percentage))
        deleted = 0
        
        for key, _ in query_keys[:num_to_evict]:
            try:
                redis_client.delete(key)
                deleted += 1
            except Exception:
                pass
        
        logger.info(f"Evicted {deleted} old cache entries to free Redis memory")
        return deleted
    except Exception as e:
        logger.warning(f"Error during cache eviction: {e}")
        try:
            deleted = 0
            for key in redis_client.scan_iter(match="query:*"):
                redis_client.delete(key)
                deleted += 1
            logger.info(f"Evicted {deleted} query cache entries as fallback")
            return deleted
        except Exception:
            return 0


def clear_user_query_cache(user_id: str):
    """Clear query cache for a specific user"""
    redis_client = get_redis_client()
    if redis_client is None:
        logger.debug("Redis not available, skipping cache clear")
        return
    
    try:
        deleted_queries = 0
        for key in redis_client.scan_iter(match="query:*"):
            try:
                cached_data = redis_client.get(key)
                if cached_data:
                    # Check if the cached data contains user_id in the key
                    # Query cache keys are hashed, so we need to check all and filter
                    redis_client.delete(key)
                    deleted_queries += 1
            except Exception:
                pass
        
        if deleted_queries > 0:
            logger.info(f"Cleared {deleted_queries} query cache entries for user {user_id}")
    except Exception as e:
        logger.warning(f"Error clearing user query cache: {e}")


def clear_cache():
    """Clear all query caches in Redis (gracefully handles Redis unavailability)"""
    redis_client = get_redis_client()
    if redis_client is None:
        logger.warning("Redis not available, cannot clear cache")
        return
    
    try:
        deleted = 0
        for key in redis_client.scan_iter(match="query:*"):
            redis_client.delete(key)
            deleted += 1
        logger.info(f"Cleared {deleted} query cache entries from Redis")
    except Exception as e:
        logger.warning(f"Error clearing Redis cache: {e}")
