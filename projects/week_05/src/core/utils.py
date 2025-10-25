"""Utility functions and classes for optimization algorithms."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Callable, Optional
import random
import time
import numpy as np


@dataclass
class Timer:
    """Simple timer for measuring elapsed time."""
    
    start: float = field(default_factory=time.time)
    
    def reset(self) -> None:
        """Reset the timer."""
        self.start = time.time()
    
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start


def rng(seed: Optional[int] = None) -> random.Random:
    """
    Create a random number generator with optional seed.
    
    Args:
        seed: Random seed (None for non-deterministic)
        
    Returns:
        Random number generator instance
    """
    r = random.Random()
    if seed is not None:
        r.seed(seed)
        np.random.seed(seed % (2**32 - 1))
    return r


@dataclass
class RunResult:
    """Result of a single algorithm run."""
    
    best_value: float
    final_value: float
    iters: int
    time_s: float
    accept_rate: Optional[float] = None
    extras: Optional[Dict[str, Any]] = None
    
    def __repr__(self) -> str:
        return (f"RunResult(best={self.best_value:.2f}, "
                f"final={self.final_value:.2f}, "
                f"iters={self.iters}, time={self.time_s:.2f}s)")


# Type alias for stop condition functions
StopFn = Callable[[int, float, Timer], bool]


def stop_by(
    time_budget_s: Optional[float] = None,
    it_budget: Optional[int] = None,
    no_improve_limit: Optional[int] = None
) -> StopFn:
    """
    Create a stopping condition function.
    
    Args:
        time_budget_s: Maximum time in seconds
        it_budget: Maximum number of iterations
        no_improve_limit: Maximum iterations without improvement
        
    Returns:
        Function that returns True when stopping condition is met
        
    Example:
        >>> timer = Timer()
        >>> stop_fn = stop_by(time_budget_s=60.0, it_budget=1000)
        >>> stop_fn(500, 123.45, timer)
        False
    """
    no_improve_counter = [0]  # Mutable to track state
    last_best = [float('inf')]
    
    def _stop(it: int, best: float, t: Timer) -> bool:
        # Check time budget
        if time_budget_s is not None and t.elapsed() >= time_budget_s:
            return True
        
        # Check iteration budget
        if it_budget is not None and it >= it_budget:
            return True
        
        # Check no-improvement limit
        if no_improve_limit is not None:
            if best < last_best[0]:
                last_best[0] = best
                no_improve_counter[0] = 0
            else:
                no_improve_counter[0] += 1
                if no_improve_counter[0] >= no_improve_limit:
                    return True
        
        return False
    
    return _stop


def format_time(seconds: float) -> str:
    """
    Format time in human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string (e.g., "2.5s", "1m 30s", "1h 5m")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    else:
        h, remainder = divmod(seconds, 3600)
        m, s = divmod(remainder, 60)
        return f"{int(h)}h {int(m)}m"
