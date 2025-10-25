"""Neighborhood structures and delta evaluation for various problems."""

from __future__ import annotations
from typing import List, Tuple, Callable
import math


# ============================================================================
# TSP (Traveling Salesman Problem)
# ============================================================================

def tour_length(tour: List[int], dist) -> float:
    """
    Calculate total tour length.
    
    Args:
        tour: Permutation of cities
        dist: Distance matrix (2D array or callable)
        
    Returns:
        Total distance
    """
    n = len(tour)
    if callable(dist):
        return sum(dist(tour[i], tour[(i+1) % n]) for i in range(n))
    else:
        return sum(dist[tour[i]][tour[(i+1) % n]] for i in range(n))


def two_opt_delta(tour: List[int], i: int, k: int, dist) -> float:
    """
    Calculate change in tour length for 2-opt move.
    
    2-opt reverses the segment tour[i:k+1].
    
    Args:
        tour: Current tour
        i: Start index of segment to reverse
        k: End index of segment to reverse (inclusive)
        dist: Distance matrix
        
    Returns:
        Delta (positive = worse, negative = better)
        
    Example:
        Tour: [0, 1, 2, 3, 4, 0]
        2-opt(i=1, k=3): [0, 3, 2, 1, 4, 0]
        Old edges: (0,1), (3,4)
        New edges: (0,3), (1,4)
    """
    n = len(tour)
    a, b = tour[i-1], tour[i]
    c, d = tour[k], tour[(k+1) % n]
    
    if callable(dist):
        before = dist(a, b) + dist(c, d)
        after = dist(a, c) + dist(b, d)
    else:
        before = dist[a][b] + dist[c][d]
        after = dist[a][c] + dist[b][d]
    
    return after - before


def apply_two_opt(tour: List[int], i: int, k: int) -> None:
    """
    Apply 2-opt move in-place (reverse segment tour[i:k+1]).
    
    Args:
        tour: Tour to modify
        i: Start index
        k: End index (inclusive)
    """
    tour[i:k+1] = reversed(tour[i:k+1])


def three_opt_moves(tour: List[int]) -> List[Tuple[int, int, int]]:
    """
    Generate all possible 3-opt moves.
    
    A 3-opt move breaks 3 edges and reconnects in a different way.
    
    Args:
        tour: Current tour
        
    Returns:
        List of (i, j, k) tuples representing possible moves
    """
    n = len(tour)
    moves = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                moves.append((i, j, k))
    return moves


# ============================================================================
# Knapsack Problem
# ============================================================================

def knapsack_value(x: List[int], values: List[float]) -> float:
    """
    Calculate total value of selected items.
    
    Args:
        x: Binary selection vector
        values: Item values
        
    Returns:
        Total value
    """
    return sum(v * sel for v, sel in zip(values, x))


def knapsack_weight(x: List[int], weights: List[float]) -> float:
    """
    Calculate total weight of selected items.
    
    Args:
        x: Binary selection vector
        weights: Item weights
        
    Returns:
        Total weight
    """
    return sum(w * sel for w, sel in zip(weights, x))


def knapsack_delta(
    x: List[int],
    flip_idx: int,
    values: List[float],
    weights: List[float],
    capacity: float,
    penalty: float = 1e6
) -> float:
    """
    Calculate delta for flipping one bit in knapsack solution.
    
    Uses penalty method: objective = -value + penalty * max(0, weight - capacity)
    
    Args:
        x: Current solution
        flip_idx: Index to flip
        values: Item values
        weights: Item weights
        capacity: Knapsack capacity
        penalty: Penalty coefficient for constraint violation
        
    Returns:
        Delta (positive = worse, negative = better)
    """
    v = values[flip_idx]
    w = weights[flip_idx]
    
    # Current state
    current_val = knapsack_value(x, values)
    current_weight = knapsack_weight(x, weights)
    current_overflow = max(0, current_weight - capacity)
    current_obj = -current_val + penalty * current_overflow
    
    # New state (after flip)
    if x[flip_idx] == 0:  # Adding item
        new_val = current_val + v
        new_weight = current_weight + w
    else:  # Removing item
        new_val = current_val - v
        new_weight = current_weight - w
    
    new_overflow = max(0, new_weight - capacity)
    new_obj = -new_val + penalty * new_overflow
    
    return new_obj - current_obj


def knapsack_repair(
    x: List[int],
    weights: List[float],
    values: List[float],
    capacity: float
) -> None:
    """
    Repair infeasible knapsack solution by removing least efficient items.
    
    Modifies x in-place.
    
    Args:
        x: Solution to repair
        weights: Item weights
        values: Item values
        capacity: Knapsack capacity
    """
    current_weight = knapsack_weight(x, weights)
    
    if current_weight <= capacity:
        return  # Already feasible
    
    # Sort selected items by value/weight ratio (ascending)
    selected = [(i, values[i] / weights[i]) for i in range(len(x)) if x[i] == 1]
    selected.sort(key=lambda t: t[1])
    
    # Remove items until feasible
    for idx, _ in selected:
        if current_weight <= capacity:
            break
        x[idx] = 0
        current_weight -= weights[idx]


# ============================================================================
# Graph Coloring
# ============================================================================

def coloring_conflicts(colors: List[int], edges: List[Tuple[int, int]]) -> int:
    """
    Count number of conflicting edges (adjacent vertices with same color).
    
    Args:
        colors: Color assignment for each vertex
        edges: List of edges (u, v)
        
    Returns:
        Number of conflicts
    """
    return sum(1 for u, v in edges if colors[u] == colors[v])


def coloring_delta(
    colors: List[int],
    vertex: int,
    new_color: int,
    neighbors: List[List[int]]
) -> int:
    """
    Calculate change in conflicts for recoloring one vertex.
    
    Args:
        colors: Current coloring
        vertex: Vertex to recolor
        new_color: New color
        neighbors: Adjacency list
        
    Returns:
        Change in number of conflicts
    """
    old_color = colors[vertex]
    if old_color == new_color:
        return 0
    
    delta = 0
    for neighbor in neighbors[vertex]:
        neighbor_color = colors[neighbor]
        
        # Old conflicts
        if neighbor_color == old_color:
            delta -= 1
        
        # New conflicts
        if neighbor_color == new_color:
            delta += 1
    
    return delta


# ============================================================================
# Max-SAT
# ============================================================================

def maxsat_satisfied(
    assignment: List[bool],
    clauses: List[List[int]]
) -> int:
    """
    Count number of satisfied clauses.
    
    Args:
        assignment: Boolean assignment to variables
        clauses: List of clauses, where each clause is list of literals
                 (positive = variable, negative = negated variable)
        
    Returns:
        Number of satisfied clauses
        
    Example:
        >>> assignment = [True, False, True]
        >>> clauses = [[1, 2], [-1, 3], [-2, -3]]  # (x1 OR x2) AND ...
        >>> maxsat_satisfied(assignment, clauses)
        2
    """
    count = 0
    for clause in clauses:
        satisfied = False
        for lit in clause:
            var_idx = abs(lit) - 1
            var_val = assignment[var_idx]
            
            if (lit > 0 and var_val) or (lit < 0 and not var_val):
                satisfied = True
                break
        
        if satisfied:
            count += 1
    
    return count


def maxsat_delta(
    assignment: List[bool],
    flip_idx: int,
    clauses: List[List[int]],
    clause_index: List[List[int]]
) -> int:
    """
    Calculate change in satisfied clauses for flipping one variable.
    
    Args:
        assignment: Current assignment
        flip_idx: Variable index to flip
        clauses: List of clauses
        clause_index: For each variable, list of clause indices containing it
        
    Returns:
        Delta (positive = more satisfied clauses)
    """
    delta = 0
    lit_positive = flip_idx + 1
    lit_negative = -(flip_idx + 1)
    
    # Check all clauses containing this variable
    for clause_idx in clause_index[flip_idx]:
        clause = clauses[clause_idx]
        
        # Check if currently satisfied
        currently_satisfied = False
        for lit in clause:
            var_idx = abs(lit) - 1
            var_val = assignment[var_idx]
            if (lit > 0 and var_val) or (lit < 0 and not var_val):
                currently_satisfied = True
                break
        
        # Check if will be satisfied after flip
        new_assignment = assignment[:]
        new_assignment[flip_idx] = not new_assignment[flip_idx]
        
        will_be_satisfied = False
        for lit in clause:
            var_idx = abs(lit) - 1
            var_val = new_assignment[var_idx]
            if (lit > 0 and var_val) or (lit < 0 and not var_val):
                will_be_satisfied = True
                break
        
        if will_be_satisfied and not currently_satisfied:
            delta += 1
        elif currently_satisfied and not will_be_satisfied:
            delta -= 1
    
    return delta
