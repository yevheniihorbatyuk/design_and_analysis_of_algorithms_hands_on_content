from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import asyncio
from datetime import datetime

class BaseExternalAPI(ABC):
    def __init__(self, name: str, rate_limit: int = 5, time_window: int = 1):
        self.name = name
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.request_timestamps: list[float] = []
        self._lock = asyncio.Lock()

    @abstractmethod
    async def search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute search request to external API
        Must be implemented by concrete providers
        """
        pass

    async def _check_rate_limit(self) -> bool:
        """
        Implement token bucket rate limiting for external API calls
        """
        async with self._lock:
            current_time = datetime.now().timestamp()
            # Remove old timestamps
            self.request_timestamps = [ts for ts in self.request_timestamps 
                                    if current_time - ts < self.time_window]
            
            if len(self.request_timestamps) < self.rate_limit:
                self.request_timestamps.append(current_time)
                return True
            return False

    async def execute_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute rate-limited request to external API
        """
        if not await self._check_rate_limit():
            raise Exception(f"Rate limit exceeded for provider {self.name}")
        
        return await self.search(params)
    