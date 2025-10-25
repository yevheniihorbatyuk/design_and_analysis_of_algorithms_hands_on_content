"""Simple problem instance generators for testing and demos."""

from __future__ import annotations
from typing import List, Tuple
import random
import math


def tsp_euclidean(n: int, seed: int = 42, width: float = 1000.0) -> List[Tuple[float, float]]:
    """
    Generate random Euclidean TSP instance.
    
    Args:
        n: Number of cities
        seed: Random seed
        width: Coordinate range [0, width)
        
    Returns:
        List of (x, y) coordinates
        
    Example:
        >>> coords = tsp_euclidean(50, seed=42)
        >>> len(coords)
        50
    """
    rng = random.Random(seed)
    return [(rng.uniform(0, width), rng.uniform(0, width)) for _ in range(n)]


def tsp_random(n: int, seed: int = 42, max_dist: float = 100.0) -> List[List[float]]:
    """
    Generate random asymmetric TSP instance (distance matrix).
    
    Args:
        n: Number of cities
        seed: Random seed
        max_dist: Maximum distance
        
    Returns:
        Distance matrix
    """
    rng = random.Random(seed)
    dist = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            d = rng.uniform(1, max_dist)
            dist[i][j] = d
            dist[j][i] = d  # Symmetric
    
    return dist


def knapsack_random(
    n: int,
    capacity_ratio: float = 0.5,
    seed: int = 42
) -> Tuple[List[float], List[float], float]:
    """
    Generate random knapsack instance.
    
    Args:
        n: Number of items
        capacity_ratio: Capacity as fraction of total weight
        seed: Random seed
        
    Returns:
        Tuple of (values, weights, capacity)
        
    Example:
        >>> values, weights, capacity = knapsack_random(100, seed=42)
        >>> len(values)
        100
    """
    rng = random.Random(seed)
    
    weights = [rng.uniform(1, 100) for _ in range(n)]
    values = [rng.uniform(1, 100) for _ in range(n)]
    capacity = sum(weights) * capacity_ratio
    
    return values, weights, capacity


def knapsack_correlated(
    n: int,
    correlation: float = 0.8,
    capacity_ratio: float = 0.5,
    seed: int = 42
) -> Tuple[List[float], List[float], float]:
    """
    Generate knapsack with correlated weights and values.
    
    Higher correlation makes problem harder.
    
    Args:
        n: Number of items
        correlation: Correlation between weight and value (0-1)
        capacity_ratio: Capacity as fraction of total weight
        seed: Random seed
        
    Returns:
        Tuple of (values, weights, capacity)
    """
    rng = random.Random(seed)
    
    weights = [rng.uniform(1, 100) for _ in range(n)]
    
    # Values correlated with weights
    values = []
    for w in weights:
        base = w * correlation
        noise = rng.uniform(0, 100) * (1 - correlation)
        values.append(base + noise)
    
    capacity = sum(weights) * capacity_ratio
    
    return values, weights, capacity


def graph_random(
    n: int,
    p: float = 0.5,
    seed: int = 42
) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Generate random graph (Erdős-Rényi).
    
    Args:
        n: Number of vertices
        p: Edge probability
        seed: Random seed
        
    Returns:
        Tuple of (n_vertices, edge_list)
        
    Example:
        >>> n, edges = graph_random(50, p=0.3, seed=42)
        >>> len(edges)  # Approximately n*(n-1)/2 * p
        367
    """
    rng = random.Random(seed)
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append((i, j))
    
    return n, edges


def maxsat_random(
    n_vars: int,
    n_clauses: int,
    k: int = 3,
    seed: int = 42
) -> List[List[int]]:
    """
    Generate random k-SAT instance.
    
    Args:
        n_vars: Number of variables
        n_clauses: Number of clauses
        k: Clause size (variables per clause)
        seed: Random seed
        
    Returns:
        List of clauses (each clause is list of literals)
        
    Example:
        >>> clauses = maxsat_random(50, 215, k=3, seed=42)
        >>> len(clauses)
        215
        >>> all(len(c) == 3 for c in clauses)
        True
        
    Note:
        Positive literal = variable, negative = negated variable
        Variables are 1-indexed: 1, 2, ..., n_vars
    """
    rng = random.Random(seed)
    clauses = []
    
    for _ in range(n_clauses):
        # Pick k distinct variables
        vars_in_clause = rng.sample(range(1, n_vars + 1), k)
        
        # Randomly negate each variable
        clause = [v if rng.random() < 0.5 else -v for v in vars_in_clause]
        clauses.append(clause)
    
    return clauses


def maxsat_planted(
    n_vars: int,
    n_clauses: int,
    k: int = 3,
    seed: int = 42
) -> Tuple[List[List[int]], List[bool]]:
    """
    Generate MAX-SAT with known solution (planted satisfying assignment).
    
    Args:
        n_vars: Number of variables
        n_clauses: Number of clauses
        k: Clause size
        seed: Random seed
        
    Returns:
        Tuple of (clauses, satisfying_assignment)
    """
    rng = random.Random(seed)
    
    # Generate random satisfying assignment
    assignment = [rng.choice([True, False]) for _ in range(n_vars)]
    
    clauses = []
    for _ in range(n_clauses):
        # Pick k distinct variables
        vars_in_clause = rng.sample(range(1, n_vars + 1), k)
        
        # Create clause that's satisfied by assignment
        clause = []
        for v in vars_in_clause:
            # Make literal consistent with assignment
            if assignment[v - 1]:
                clause.append(v if rng.random() < 0.8 else -v)
            else:
                clause.append(-v if rng.random() < 0.8 else v)
        
        clauses.append(clause)
    
    return clauses, assignment


def tsp_clustered(
    n_clusters: int,
    points_per_cluster: int,
    cluster_spread: float = 50.0,
    cluster_distance: float = 500.0,
    seed: int = 42
) -> List[Tuple[float, float]]:
    """
    Generate clustered TSP instance (harder than random).
    
    Args:
        n_clusters: Number of clusters
        points_per_cluster: Points in each cluster
        cluster_spread: Spread within cluster
        cluster_distance: Distance between cluster centers
        seed: Random seed
        
    Returns:
        List of (x, y) coordinates
    """
    rng = random.Random(seed)
    coords = []
    
    # Generate cluster centers on a circle
    for i in range(n_clusters):
        angle = 2 * math.pi * i / n_clusters
        center_x = cluster_distance * math.cos(angle)
        center_y = cluster_distance * math.sin(angle)
        
        # Generate points around this center
        for _ in range(points_per_cluster):
            offset_x = rng.gauss(0, cluster_spread)
            offset_y = rng.gauss(0, cluster_spread)
            coords.append((center_x + offset_x, center_y + offset_y))
    
    return coords
