from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Literal, List, Tuple, Dict, Any
from src.caches import (
    BaseCache,
    RedisCache,
    MemcachedCache,
    MultiLevelCache,
    MemoryCache
)
from src.rate_limit import (
    BaseRateLimiter,
    InMemoryRateLimiter,
    RedisRateLimiter,
    TokenBucketRateLimiter,
    SlidingWindowRateLimiter,
    DistributedRateLimiter,
    AdaptiveRateLimiter,
    HierarchicalRateLimiter
)

import redis
import memcache

class Settings(BaseSettings):

    APP_NAME: str = "Service Aggregator API"

    # Cache settings
    CACHE_TYPE: Literal["memory", "redis", "memcached", "multi"] = "redis"
    CACHE_STRATEGY: Literal["lru", "ttl"] = "lru"
    CACHE_SIZE: int = 10
    CACHE_TTL: int = 3600
    
    # Redis settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    # Memcached settings
    MEMCACHED_SERVERS: list[str] = ["localhost:11211"]
    
    # Rate limit settings
    RATE_LIMIT_TYPE: Literal["memory", "redis", "token"] = "token"
    RATE_LIMIT_WINDOW: int = 10
    RATE_LIMIT_REQUESTS: int = 100
    TOKEN_BUCKET_CAPACITY: int = 5
    TOKEN_BUCKET_REFILL_RATE: float = 25.0

    # Distributed rate limit settings
    REDIS_CLUSTER_NODES: List[Dict[str, Any]] = [
        {"host": "localhost", "port": 6379}
    ]


    # Adaptive rate limit settings
    ADAPTIVE_BASE_WINDOW: int = 60
    ADAPTIVE_BASE_REQUESTS: int = 1000
    ADAPTIVE_MIN_WINDOW: int = 1
    ADAPTIVE_MAX_WINDOW: int = 3600
    
    # Hierarchical rate limit settings
    HIERARCHICAL_LIMITS: List[Tuple[str, int, int]] = [
        ("user", 60, 100),
        ("ip", 60, 1000),
        ("global", 60, 10000)
    ]
    
    # API Provider settings
    WEATHER_API_KEY: str = ""
    NEWS_API_KEY: str = ""
    TRANSLATION_API_KEY: str = ""

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()

def get_cache() -> BaseCache:
    settings = get_settings()
    
    if settings.CACHE_TYPE == "redis":
        return RedisCache(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            ttl_seconds=settings.CACHE_TTL
        )
    elif settings.CACHE_TYPE == "memcached":
        return MemcachedCache(
            servers=settings.MEMCACHED_SERVERS,
            ttl_seconds=settings.CACHE_TTL
        )
    elif settings.CACHE_TYPE == "multi":
        return MultiLevelCache([
            MemoryCache(settings.CACHE_STRATEGY, settings.CACHE_SIZE, settings.CACHE_TTL),
            RedisCache(ttl_seconds=settings.CACHE_TTL)
        ])
    else:
        return MemoryCache(
            strategy=settings.CACHE_STRATEGY,
            capacity=settings.CACHE_SIZE,
            ttl_seconds=settings.CACHE_TTL
        )

def get_rate_limiter() -> BaseRateLimiter:
    settings = get_settings()
    
    if settings.RATE_LIMIT_TYPE == "redis":
        redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB
        )
        return RedisRateLimiter(
            redis_client,
            settings.RATE_LIMIT_WINDOW,
            settings.RATE_LIMIT_REQUESTS
        )
    elif settings.RATE_LIMIT_TYPE == "token":
        return TokenBucketRateLimiter(
            capacity=settings.TOKEN_BUCKET_CAPACITY,
            refill_rate=settings.TOKEN_BUCKET_REFILL_RATE
        )
    elif settings.RATE_LIMIT_TYPE == "distributed":
        return DistributedRateLimiter(
            redis_nodes=settings.REDIS_CLUSTER_NODES,
            window_size=settings.RATE_LIMIT_WINDOW,
            max_requests=settings.RATE_LIMIT_REQUESTS
        )
    elif settings.RATE_LIMIT_TYPE == "adaptive":
        return AdaptiveRateLimiter(
            base_window=settings.ADAPTIVE_BASE_WINDOW,
            base_max_requests=settings.ADAPTIVE_BASE_REQUESTS,
            min_window=settings.ADAPTIVE_MIN_WINDOW,
            max_window=settings.ADAPTIVE_MAX_WINDOW
        )
    elif settings.RATE_LIMIT_TYPE == "hierarchical":
        return HierarchicalRateLimiter(
            limits=settings.HIERARCHICAL_LIMITS
        )
    else:  # "memory"
        return InMemoryRateLimiter(
            window_size=settings.RATE_LIMIT_WINDOW,
            max_requests=settings.RATE_LIMIT_REQUESTS
        )