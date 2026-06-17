"""
Advanced caching strategies for OASIS Agentic Pipeline.

Implements multi-layered caching with Redis backend, in-memory fallback,
and intelligent cache invalidation strategies.
"""

import time
import json
import hashlib
import logging
from typing import Any, Optional, Dict, List, Callable
from functools import wraps
from dataclasses import dataclass
import threading

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available, using in-memory cache only")


logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    value: Any
    timestamp: float
    ttl: int
    hit_count: int = 0
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return time.time() - self.timestamp > self.ttl

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "value": self.value,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
            "hit_count": self.hit_count,
            "size_bytes": self.size_bytes,
            "expired": self.is_expired(),
        }


class RedisCache:
    """Redis-based distributed cache."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl: int = 3600,
        key_prefix: str = "oasis:",
    ):
        """
        Initialize Redis cache.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            default_ttl: Default time-to-live in seconds
            key_prefix: Prefix for all cache keys
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis package not available")

        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix

        self.client = None
        self._connect()

    def _connect(self):
        """Establish Redis connection."""
        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            # Test connection
            self.client.ping()
            logger.info(f"Redis cache connected to {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.client = None

    def _make_key(self, key: str) -> str:
        """Create prefixed cache key."""
        return f"{self.key_prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from Redis cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        if not self.client:
            return None

        try:
            prefixed_key = self._make_key(key)
            data = self.client.get(prefixed_key)

            if data is None:
                return None

            # Deserialize. JSON (not pickle) — never deserialize untrusted data
            # from a shared/poisonable store with pickle (arbitrary code execution).
            return json.loads(data)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in Redis cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if not specified)

        Returns:
            True if successful
        """
        if not self.client:
            return False

        try:
            prefixed_key = self._make_key(key)
            ttl = ttl or self.default_ttl

            # Serialize as JSON (safe to deserialize). Non-JSON-serializable values
            # raise TypeError here and are handled by the surrounding except ->
            # the value is simply not cached (caller recomputes).
            data = json.dumps(value).encode("utf-8")

            # Set with expiration
            self.client.setex(prefixed_key, ttl, data)
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        Delete key from Redis cache.

        Args:
            key: Cache key

        Returns:
            True if successful
        """
        if not self.client:
            return False

        try:
            prefixed_key = self._make_key(key)
            self.client.delete(prefixed_key)
            return True
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False

    def clear(self) -> bool:
        """
        Clear all Oasis cache keys from Redis.

        Returns:
            True if successful
        """
        if not self.client:
            return False

        try:
            # Delete all keys with prefix
            pattern = f"{self.key_prefix}*"
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get Redis cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        if not self.client:
            return {"status": "disconnected"}

        try:
            info = self.client.info()
            pattern = f"{self.key_prefix}*"
            keys = self.client.keys(pattern)

            return {
                "status": "connected",
                "total_keys": len(keys),
                "memory_used": info.get("used_memory_human", "N/A"),
                "hit_rate": info.get("keyspace_hits", 0),
                "miss_rate": info.get("keyspace_misses", 0),
            }
        except Exception as e:
            logger.error(f"Redis stats error: {e}")
            return {"status": "error", "error": str(e)}


class InMemoryCache:
    """High-performance in-memory cache with LRU eviction."""

    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        """
        Initialize in-memory cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.lock = threading.RLock()

        logger.info(f"In-memory cache initialized (max_size={max_size}, ttl={default_ttl}s)")

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        with self.lock:
            if key not in self.cache:
                return None

            entry = self.cache[key]

            # Check if expired
            if entry.is_expired():
                del self.cache[key]
                self.access_order.remove(key)
                return None

            # Update hit count and access order
            entry.hit_count += 1
            self.access_order.remove(key)
            self.access_order.append(key)

            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds

        Returns:
            True if successful
        """
        with self.lock:
            # Calculate size (best-effort; in-memory cache stores the live object,
            # so this is only for LRU accounting — no deserialization happens).
            try:
                size = len(json.dumps(value).encode("utf-8"))
            except (TypeError, ValueError):
                size = len(repr(value).encode("utf-8"))

            ttl = ttl or self.default_ttl

            # Evict if necessary
            if key not in self.cache and len(self.cache) >= self.max_size:
                self._evict_lru()

            # Create entry
            entry = CacheEntry(value=value, timestamp=time.time(), ttl=ttl, size_bytes=size)

            self.cache[key] = entry
            self.access_order.append(key)

            return True

    def delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if successful
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.access_order.remove(key)
                return True
            return False

    def clear(self) -> bool:
        """
        Clear all cache entries.

        Returns:
            True if successful
        """
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            return True

    def _evict_lru(self):
        """Evict least recently used entry."""
        if self.access_order:
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            total_size = sum(entry.size_bytes for entry in self.cache.values())
            total_hits = sum(entry.hit_count for entry in self.cache.values())

            return {
                "status": "connected",
                "total_keys": len(self.cache),
                "max_size": self.max_size,
                "utilization": len(self.cache) / self.max_size,
                "total_size_bytes": total_size,
                "total_hits": total_hits,
                "avg_hit_count": total_hits / len(self.cache) if self.cache else 0,
            }


class MultiLayerCache:
    """
    Multi-layered cache with Redis and in-memory fallback.

    Cache hierarchy:
    1. In-memory cache (fastest)
    2. Redis cache (distributed)
    3. Cache miss (compute and cache)
    """

    def __init__(
        self,
        use_redis: bool = True,
        redis_config: Optional[Dict[str, Any]] = None,
        in_memory_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize multi-layer cache.

        Args:
            use_redis: Enable Redis backend
            redis_config: Redis configuration
            in_memory_config: In-memory cache configuration
        """
        self.use_redis = use_redis and REDIS_AVAILABLE
        self.in_memory_config = in_memory_config or {"max_size": 10000, "default_ttl": 3600}

        # Initialize caches
        self.memory_cache = InMemoryCache(**self.in_memory_config)

        if self.use_redis:
            redis_config = redis_config or {}
            try:
                self.redis_cache = RedisCache(**redis_config)
                logger.info("Multi-layer cache initialized with Redis backend")
            except Exception as e:
                logger.warning(f"Redis initialization failed, using in-memory only: {e}")
                self.redis_cache = None
                self.use_redis = False
        else:
            self.redis_cache = None
            logger.info("Multi-layer cache initialized (in-memory only)")

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache (checks memory first, then Redis).

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        # Check memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value

        # Check Redis cache
        if self.use_redis and self.redis_cache:
            value = self.redis_cache.get(key)
            if value is not None:
                # Populate memory cache
                self.memory_cache.set(key, value)
                return value

        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in both cache layers.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds

        Returns:
            True if successful
        """
        # Set in memory cache
        memory_success = self.memory_cache.set(key, value, ttl)

        # Set in Redis cache
        redis_success = True
        if self.use_redis and self.redis_cache:
            redis_success = self.redis_cache.set(key, value, ttl)

        return memory_success and redis_success

    def delete(self, key: str) -> bool:
        """
        Delete key from both cache layers.

        Args:
            key: Cache key

        Returns:
            True if successful
        """
        memory_success = self.memory_cache.delete(key)
        redis_success = True
        if self.use_redis and self.redis_cache:
            redis_success = self.redis_cache.delete(key)

        return memory_success and redis_success

    def clear(self) -> bool:
        """
        Clear both cache layers.

        Returns:
            True if successful
        """
        memory_success = self.memory_cache.clear()
        redis_success = True
        if self.use_redis and self.redis_cache:
            redis_success = self.redis_cache.clear()

        return memory_success and redis_success

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics for both cache layers.

        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "memory": self.memory_cache.get_stats(),
            "redis": self.redis_cache.get_stats()
            if self.use_redis and self.redis_cache
            else {"status": "disabled"},
        }
        return stats


def cached(ttl: int = 3600, key_prefix: str = "", cache_instance: Optional[MultiLayerCache] = None):
    """
    Decorator for caching function results.

    Args:
        ttl: Time-to-live in seconds
        key_prefix: Prefix for cache keys
        cache_instance: Cache instance to use (uses global if not specified)

    Returns:
        Decorated function with caching

    Example:
        @cached(ttl=1800, key_prefix="vision_")
        def process_image(image_path):
            # Expensive processing
            return result
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix, func.__name__]

            # Add arguments to key
            if args:
                key_parts.extend(str(arg) for arg in args)
            if kwargs:
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))

            key = "_".join(key_parts)

            # Use provided cache or global instance
            cache = cache_instance or global_cache

            # Check cache
            cached_value = cache.get(key)
            if cached_value is not None:
                return cached_value

            # Execute function
            result = func(*args, **kwargs)

            # Cache result
            cache.set(key, result, ttl)

            return result

        return wrapper

    return decorator


def cache_key_generator(*args, **kwargs) -> str:
    """
    Generate consistent cache key from function arguments.

    Args:
        *args: Function positional arguments
        **kwargs: Function keyword arguments

    Returns:
        Cache key string
    """
    # Convert arguments to hash
    key_data = {"args": args, "kwargs": kwargs}
    key_string = json.dumps(key_data, sort_keys=True, default=str)
    key_hash = hashlib.md5(key_string.encode()).hexdigest()

    return key_hash


class QueryCache:
    """Specialized cache for database and API queries."""

    def __init__(self, cache: MultiLayerCache):
        """
        Initialize query cache.

        Args:
            cache: Cache instance
        """
        self.cache = cache

    def get_query_result(
        self, query_type: str, query_params: Dict[str, Any], execute_func: Callable, ttl: int = 300
    ) -> Any:
        """
        Get query result with caching.

        Args:
            query_type: Type of query (e.g., "patient_data", "biomarker_stats")
            query_params: Query parameters
            execute_func: Function to execute if cache miss
            ttl: Cache TTL in seconds

        Returns:
            Query result
        """
        # Generate cache key
        key = f"query_{query_type}_{cache_key_generator(**query_params)}"

        # Check cache
        result = self.cache.get(key)
        if result is not None:
            logger.debug(f"Query cache hit: {query_type}")
            return result

        # Execute query
        logger.debug(f"Query cache miss: {query_type}")
        result = execute_func()

        # Cache result
        self.cache.set(key, result, ttl)

        return result

    def invalidate_query_type(self, query_type: str):
        """
        Invalidate all cached queries of a type.

        Args:
            query_type: Query type to invalidate
        """
        # This would require pattern matching in Redis
        # For now, we'll just log the intent
        logger.info(f"Invalidating query cache for type: {query_type}")


# Global cache instance
global_cache = MultiLayerCache(
    use_redis=False,  # Default to in-memory only
    in_memory_config={"max_size": 5000, "default_ttl": 1800},
)

# Global query cache
query_cache = QueryCache(global_cache)


if __name__ == "__main__":
    # Test caching strategies
    print("Testing advanced caching strategies...")

    # Test in-memory cache
    print("\n1. Testing in-memory cache...")
    memory_cache = InMemoryCache(max_size=5, default_ttl=60)

    memory_cache.set("key1", "value1")
    memory_cache.set("key2", "value2")

    print(f"Get key1: {memory_cache.get('key1')}")
    print(f"Get key3: {memory_cache.get('key3')}")
    print(f"Memory cache stats: {memory_cache.get_stats()}")

    # Test multi-layer cache
    print("\n2. Testing multi-layer cache...")
    multi_cache = MultiLayerCache(use_redis=False)

    multi_cache.set("test_key", {"data": "test_value"}, ttl=30)
    result = multi_cache.get("test_key")
    print(f"Multi-layer cache result: {result}")
    print(f"Multi-layer cache stats: {multi_cache.get_stats()}")

    # Test cached decorator
    print("\n3. Testing cached decorator...")

    @cached(ttl=60, key_prefix="test_")
    def expensive_computation(n: int):
        time.sleep(0.1)
        return n * n

    # First call (not cached)
    start = time.time()
    result1 = expensive_computation(5)
    time1 = time.time() - start

    # Second call (cached)
    start = time.time()
    result2 = expensive_computation(5)
    time2 = time.time() - start

    print(f"First call: {time1:.4f}s, Second call: {time2:.4f}s")
    print(f"Cache speedup: {time1 / time2:.2f}x")

    # Test query cache
    print("\n4. Testing query cache...")

    def execute_query():
        time.sleep(0.05)
        return {"patient_id": "OAS2_0001", "mmse": 24.0}

    result = query_cache.get_query_result(
        query_type="patient_data",
        query_params={"patient_id": "OAS2_0001"},
        execute_func=execute_query,
        ttl=60,
    )
    print(f"Query cache result: {result}")

    print("\nAdvanced caching strategies test complete!")
