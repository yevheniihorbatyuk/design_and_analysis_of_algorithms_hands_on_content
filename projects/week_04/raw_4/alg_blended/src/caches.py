from abc import ABC, abstractmethod
from typing import Any, Optional, Union
from datetime import datetime, timedelta
import json
import redis
import memcache
from cachetools import TTLCache, LRUCache as CacheToolsLRU

class BaseCache(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass
        
    @abstractmethod
    def put(self, key: str, value: Any) -> None:
        pass
        
    @abstractmethod
    def delete(self, key: str) -> None:
        pass
        
    @abstractmethod
    def clear(self) -> None:
        pass

class MemoryCache(BaseCache):
    def __init__(self, strategy: str = "lru", capacity: int = 1000, ttl_seconds: int = 3600):
        self.strategy = strategy
        if strategy == "lru":
            # LRUCache doesn't support ttl directly
            self.cache = CacheToolsLRU(maxsize=capacity)
        else:
            # TTLCache does support ttl
            self.cache = TTLCache(maxsize=capacity, ttl=ttl_seconds)
            
    def get(self, key: str) -> Optional[Any]:
        try:
            return self.cache[key]
        except KeyError:
            return None
            
    def put(self, key: str, value: Any) -> None:
        self.cache[key] = value
        
    def delete(self, key: str) -> None:
        if key in self.cache:
            del self.cache[key]
            
    def clear(self) -> None:
        self.cache.clear()

class RedisCache(BaseCache):
    def __init__(self, host: str = 'localhost', port: int = 6379, 
                 db: int = 0, ttl_seconds: int = 3600):
        self.redis = redis.Redis(host=host, port=port, db=db, 
                               decode_responses=True)
        self.ttl = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        data = self.redis.get(key)
        return json.loads(data) if data else None

    def put(self, key: str, value: Any) -> None:
        self.redis.setex(key, self.ttl, json.dumps(value))
        
    def delete(self, key: str) -> None:
        self.redis.delete(key)
        
    def clear(self) -> None:
        self.redis.flushdb()

class MemcachedCache(BaseCache):
    def __init__(self, servers: list[str], ttl_seconds: int = 3600):
        self.client = memcache.Client(servers)
        self.ttl = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        return self.client.get(key)

    def put(self, key: str, value: Any) -> None:
        self.client.set(key, value, self.ttl)
        
    def delete(self, key: str) -> None:
        self.client.delete(key)
        
    def clear(self) -> None:
        self.client.flush_all()

class MultiLevelCache(BaseCache):
    def __init__(self, caches: list[BaseCache]):
        self.caches = caches

    def get(self, key: str) -> Optional[Any]:
        for i, cache in enumerate(self.caches):
            value = cache.get(key)
            if value is not None:
                # Update higher level caches
                for j in range(i):
                    self.caches[j].put(key, value)
                return value
        return None

    def put(self, key: str, value: Any) -> None:
        for cache in self.caches:
            cache.put(key, value)
            
    def delete(self, key: str) -> None:
        for cache in self.caches:
            cache.delete(key)
            
    def clear(self) -> None:
        for cache in self.caches:
            cache.clear()

