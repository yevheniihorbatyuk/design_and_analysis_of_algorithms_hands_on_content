from abc import ABC, abstractmethod
from typing import Optional, Deque, List, Dict, Any, Tuple
import redis
from datetime import datetime, timedelta
from collections import deque

class BaseRateLimiter(ABC):
    @abstractmethod
    def is_allowed(self, key: str) -> bool:
        pass

class InMemoryRateLimiter(BaseRateLimiter):
    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests: dict[str, deque[datetime]] = {}

    def is_allowed(self, key: str) -> bool:
        if key not in self.requests:
            self.requests[key] = deque()

        now = datetime.now()
        window_start = now - timedelta(seconds=self.window_size)

        while self.requests[key] and self.requests[key][0] < window_start:
            self.requests[key].popleft()

        if len(self.requests[key]) < self.max_requests:
            self.requests[key].append(now)
            return True
        return False

class RedisRateLimiter(BaseRateLimiter):
    def __init__(self, redis_client: redis.Redis, window_size: int, max_requests: int):
        self.redis = redis_client
        self.window_size = window_size
        self.max_requests = max_requests

    def is_allowed(self, key: str) -> bool:
        pipe = self.redis.pipeline()
        now = datetime.now().timestamp()
        window_start = now - self.window_size
        
        # Remove old requests
        pipe.zremrangebyscore(key, 0, window_start)
        # Count requests in window
        pipe.zcard(key)
        # Add new request
        pipe.zadd(key, {str(now): now})
        # Set expiry
        pipe.expire(key, self.window_size)
        
        results = pipe.execute()
        request_count = results[1]
        
        return request_count < self.max_requests

class TokenBucketRateLimiter(BaseRateLimiter):
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens: dict[str, float] = {}
        self.last_update: dict[str, datetime] = {}

    def is_allowed(self, key: str) -> bool:
        now = datetime.now()
        
        if key not in self.tokens:
            self.tokens[key] = self.capacity
            self.last_update[key] = now
            return True
            
        # Refill tokens
        elapsed = (now - self.last_update[key]).total_seconds()
        self.tokens[key] = min(
            self.capacity,
            self.tokens[key] + elapsed * self.refill_rate
        )
        self.last_update[key] = now
        
        if self.tokens[key] >= 1:
            self.tokens[key] -= 1
            return True
        return False

class SlidingWindowRateLimiter(BaseRateLimiter):
    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests: dict[str, Deque[datetime]] = {}

    def is_allowed(self, key: str) -> bool:
        if key not in self.requests:
            self.requests[key] = deque()

        now = datetime.now()
        window_start = now - timedelta(seconds=self.window_size)

        while self.requests[key] and self.requests[key][0] < window_start:
            self.requests[key].popleft()

        if len(self.requests[key]) < self.max_requests:
            self.requests[key].append(now)
            return True

        return False
    

class DistributedRateLimiter(BaseRateLimiter):
    """Distributed rate limiter using Redis Cluster"""
    def __init__(self, 
                 redis_nodes: List[Dict[str, Any]],
                 window_size: int,
                 max_requests: int):
        self.window_size = window_size
        self.max_requests = max_requests
        self.redis_cluster = RedisCluster(
            startup_nodes=redis_nodes,
            decode_responses=True
        )

    def is_allowed(self, key: str) -> bool:
        lua_script = """
        local key = KEYS[1]
        local window = tonumber(ARGV[1])
        local max_requests = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        
        -- Remove old requests
        redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
        
        -- Count requests in current window
        local count = redis.call('ZCARD', key)
        
        if count < max_requests then
            -- Add new request
            redis.call('ZADD', key, now, now)
            -- Set expiry
            redis.call('EXPIRE', key, window)
            return 1
        end
        
        return 0
        """
        
        now = datetime.now().timestamp()
        result = self.redis_cluster.eval(
            lua_script,
            1,  # number of keys
            key,  # key
            self.window_size,
            self.max_requests,
            now
        )
        
        return bool(result)

class AdaptiveRateLimiter(BaseRateLimiter):
    """Rate limiter that adapts limits based on system load"""
    def __init__(self, 
                 base_window: int,
                 base_max_requests: int,
                 min_window: int = 1,
                 max_window: int = 3600):
        self.base_window = base_window
        self.base_max_requests = base_max_requests
        self.min_window = min_window
        self.max_window = max_window
        self.requests = {}
        self.load_history = deque(maxlen=10)
        self.last_adjustment = datetime.now()

    def _get_system_load(self) -> float:
        """Get current system load (CPU usage)"""
        return psutil.cpu_percent() / 100.0

    def _adjust_window(self) -> None:
        """Adjust window size based on system load"""
        now = datetime.now()
        if (now - self.last_adjustment).total_seconds() < 60:
            return

        current_load = self._get_system_load()
        self.load_history.append(current_load)
        avg_load = sum(self.load_history) / len(self.load_history)

        # Adjust window based on load
        if avg_load > 0.8:  # High load
            self.base_window = min(
                self.base_window * 1.5,
                self.max_window
            )
        elif avg_load < 0.2:  # Low load
            self.base_window = max(
                self.base_window * 0.8,
                self.min_window
            )

        self.last_adjustment = now

    def is_allowed(self, key: str) -> bool:
        self._adjust_window()
        
        if key not in self.requests:
            self.requests[key] = deque()

        now = datetime.now()
        window_start = now - timedelta(seconds=self.base_window)

        # Clear old requests
        while self.requests[key] and self.requests[key][0] < window_start:
            self.requests[key].popleft()

        if len(self.requests[key]) < self.base_max_requests:
            self.requests[key].append(now)
            return True

        return False

class HierarchicalRateLimiter(BaseRateLimiter):
    """Rate limiter with multiple levels (e.g., per-user, per-IP, global)"""
    def __init__(self, limits: List[Tuple[str, int, int]]):
        """
        limits: List of (level_name, window_size, max_requests)
        e.g., [("user", 60, 100), ("ip", 60, 1000), ("global", 60, 10000)]
        """
        self.limiters = {
            level: InMemoryRateLimiter(window, max_req)
            for level, window, max_req in limits
        }

    def is_allowed(self, key: str, context: Dict[str, str]) -> bool:
        """
        Check all levels of rate limiting
        context: Dictionary with keys for each level (e.g., {"user": "user123", "ip": "1.2.3.4"})
        """
        for level, limiter in self.limiters.items():
            if level in context:
                if not limiter.is_allowed(context[level]):
                    return False
        return True