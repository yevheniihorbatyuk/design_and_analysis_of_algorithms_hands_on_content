"""Metrics and statistical analysis for optimization results."""

from __future__ import annotations
from typing import List, Dict, Optional
import numpy as np


def summarize(values: List[float]) -> Dict[str, float]:
    """
    Compute summary statistics for a list of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        Dictionary with statistics: best, worst, median, p95, mean, std
        
    Example:
        >>> results = [100.5, 95.3, 102.1, 98.7, 101.2]
        >>> stats = summarize(results)
        >>> print(f"Best: {stats['best']:.2f}")
        Best: 95.30
    """
    if not values:
        return {
            "best": float('inf'),
            "worst": float('inf'),
            "median": float('inf'),
            "p95": float('inf'),
            "mean": float('inf'),
            "std": 0.0,
        }
    
    arr = np.array(values, dtype=float)
    return {
        "best": float(arr.min()),
        "worst": float(arr.max()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def gap_to_optimal(value: float, optimal: float) -> float:
    """
    Calculate percentage gap to optimal solution.
    
    Args:
        value: Current solution value
        optimal: Optimal solution value
        
    Returns:
        Gap in percentage (0.0 = optimal)
        
    Example:
        >>> gap_to_optimal(105.0, 100.0)
        5.0
    """
    if optimal == 0:
        return float('inf') if value != 0 else 0.0
    return abs(value - optimal) / abs(optimal) * 100.0


def improvement_ratio(initial: float, final: float) -> float:
    """
    Calculate improvement ratio (how much better is final vs initial).
    
    Args:
        initial: Initial solution value (minimization)
        final: Final solution value (minimization)
        
    Returns:
        Improvement ratio (positive = improvement)
        
    Example:
        >>> improvement_ratio(100.0, 80.0)  # 20% improvement
        0.2
    """
    if initial == 0:
        return 0.0
    return (initial - final) / abs(initial)


def confidence_interval(
    values: List[float],
    confidence: float = 0.95
) -> tuple[float, float]:
    """
    Calculate confidence interval using t-distribution.
    
    Args:
        values: List of measurements
        confidence: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
        
    Example:
        >>> results = [100, 102, 98, 101, 99]
        >>> lower, upper = confidence_interval(results)
        >>> print(f"95% CI: [{lower:.2f}, {upper:.2f}]")
    """
    if len(values) < 2:
        mean = np.mean(values) if values else 0.0
        return (mean, mean)
    
    from scipy import stats
    arr = np.array(values)
    mean = arr.mean()
    se = stats.sem(arr)
    margin = se * stats.t.ppf((1 + confidence) / 2, len(arr) - 1)
    return (mean - margin, mean + margin)


def convergence_speed(
    trajectory: List[float],
    target_gap: float = 0.01
) -> Optional[int]:
    """
    Find iteration where solution reaches within target_gap of best.
    
    Args:
        trajectory: List of objective values over iterations
        target_gap: Acceptable gap (default 1%)
        
    Returns:
        Iteration number or None if target not reached
        
    Example:
        >>> trajectory = [1000, 950, 920, 905, 901, 900, 900]
        >>> convergence_speed(trajectory, target_gap=0.01)
        4
    """
    if not trajectory:
        return None
    
    best = min(trajectory)
    threshold = best * (1 + target_gap)
    
    for i, val in enumerate(trajectory):
        if val <= threshold:
            return i
    
    return None


def statistical_significance(
    values1: List[float],
    values2: List[float],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform statistical test to compare two algorithms.
    
    Uses Mann-Whitney U test (non-parametric).
    
    Args:
        values1: Results from algorithm 1
        values2: Results from algorithm 2
        alpha: Significance level (default 0.05)
        
    Returns:
        Dictionary with test results
        
    Example:
        >>> algo1_results = [100, 102, 98, 101]
        >>> algo2_results = [95, 97, 93, 96]
        >>> test = statistical_significance(algo1_results, algo2_results)
        >>> if test['significant']:
        ...     print(f"Algorithm 2 is better (p={test['p_value']:.4f})")
    """
    from scipy import stats
    
    statistic, p_value = stats.mannwhitneyu(
        values1, values2, alternative='two-sided'
    )
    
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "alpha": alpha,
        "n1": len(values1),
        "n2": len(values2),
    }
