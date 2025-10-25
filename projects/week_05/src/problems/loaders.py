"""Loaders for standard benchmark datasets (TSPLIB, DIMACS, SATLIB, OR-Library)."""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import re


def load_tsplib(filepath: str) -> Dict[str, Any]:
    """
    Load TSPLIB format TSP instance.
    
    Supports EUC_2D (Euclidean 2D coordinates).
    
    Args:
        filepath: Path to .tsp file
        
    Returns:
        Dictionary with 'name', 'dimension', 'coords', 'optimal' (if known)
        
    Example:
        >>> instance = load_tsplib('data/tsp/att48.tsp')
        >>> print(f"Instance: {instance['name']}, n={instance['dimension']}")
        >>> coords = instance['coords']
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    metadata = {}
    coord_section_start = -1
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        if line.startswith('NAME'):
            metadata['name'] = line.split(':')[-1].strip()
        elif line.startswith('DIMENSION'):
            metadata['dimension'] = int(line.split(':')[-1].strip())
        elif line.startswith('EDGE_WEIGHT_TYPE'):
            metadata['edge_weight_type'] = line.split(':')[-1].strip()
        elif line.startswith('NODE_COORD_SECTION'):
            coord_section_start = i + 1
            break
    
    # Parse coordinates
    coords = []
    if coord_section_start > 0:
        for i in range(coord_section_start, len(lines)):
            line = lines[i].strip()
            if line in ['EOF', 'EOF\n'] or not line:
                break
            
            parts = line.split()
            if len(parts) >= 3:
                idx = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                coords.append((x, y))
    
    metadata['coords'] = coords
    
    # Try to find optimal solution (common naming: att48.opt.tour)
    opt_file = path.with_suffix('.opt.tour')
    if opt_file.exists():
        with open(opt_file, 'r') as f:
            opt_lines = f.readlines()
        
        for line in opt_lines:
            if line.startswith('COMMENT') and 'Length' in line:
                # Extract optimal length
                match = re.search(r'(\d+)', line)
                if match:
                    metadata['optimal'] = int(match.group(1))
    
    return metadata


def load_tsplib_simple(filepath: str) -> List[Tuple[float, float]]:
    """
    Load TSPLIB and return just coordinates.
    
    Args:
        filepath: Path to .tsp file
        
    Returns:
        List of (x, y) coordinates
    """
    data = load_tsplib(filepath)
    return data.get('coords', [])


def load_dimacs_graph(filepath: str) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Load DIMACS graph format (.col file for graph coloring).
    
    Format:
        p edge n m   (n=vertices, m=edges)
        e u v        (edge from u to v)
    
    Args:
        filepath: Path to .col file
        
    Returns:
        Tuple of (n_vertices, edge_list)
        
    Example:
        >>> n, edges = load_dimacs_graph('data/coloring/myciel3.col')
        >>> print(f"Graph: {n} vertices, {len(edges)} edges")
    """
    n_vertices = 0
    edges = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            if not line or line.startswith('c'):
                continue  # Comment
            
            if line.startswith('p'):
                # Problem line: p edge n m
                parts = line.split()
                n_vertices = int(parts[2])
            
            elif line.startswith('e'):
                # Edge line: e u v
                parts = line.split()
                u = int(parts[1]) - 1  # Convert to 0-indexed
                v = int(parts[2]) - 1
                edges.append((u, v))
    
    return n_vertices, edges


def load_dimacs_cnf(filepath: str) -> Tuple[int, int, List[List[int]]]:
    """
    Load DIMACS CNF format (SAT/MaxSAT).
    
    Format:
        p cnf nvars nclauses
        clause: literal literal ... 0
    
    Args:
        filepath: Path to .cnf file
        
    Returns:
        Tuple of (n_vars, n_clauses, clauses)
        
    Example:
        >>> nvars, nclauses, clauses = load_dimacs_cnf('data/maxsat/uf50-01.cnf')
        >>> print(f"Instance: {nvars} variables, {nclauses} clauses")
    """
    n_vars = 0
    n_clauses = 0
    clauses = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            if not line or line.startswith('c'):
                continue  # Comment
            
            if line.startswith('p'):
                # Problem line: p cnf nvars nclauses
                parts = line.split()
                n_vars = int(parts[2])
                n_clauses = int(parts[3])
            
            elif line.startswith('%'):
                break  # End of clauses
            
            else:
                # Clause line
                literals = [int(x) for x in line.split()]
                if literals and literals[-1] == 0:
                    literals = literals[:-1]  # Remove trailing 0
                
                if literals:
                    clauses.append(literals)
    
    return n_vars, n_clauses, clauses


def load_knapsack(filepath: str) -> Tuple[List[float], List[float], float]:
    """
    Load knapsack instance from simple format.
    
    Format:
        n capacity
        value1 weight1
        value2 weight2
        ...
    
    Args:
        filepath: Path to knapsack file
        
    Returns:
        Tuple of (values, weights, capacity)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # First line: n capacity
    n, capacity = map(float, lines[0].strip().split())
    n = int(n)
    
    values = []
    weights = []
    
    for i in range(1, n + 1):
        if i < len(lines):
            v, w = map(float, lines[i].strip().split())
            values.append(v)
            weights.append(w)
    
    return values, weights, capacity


def load_knapsack_orlib(filepath: str) -> Tuple[List[float], List[float], float, Optional[float]]:
    """
    Load OR-Library multidimensional knapsack format.
    
    Args:
        filepath: Path to OR-Library file
        
    Returns:
        Tuple of (values, weights, capacity, optimal_value)
    """
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # First line: n (num items)
    n = int(lines[0])
    
    # Second line: profits/values
    values = list(map(float, lines[1].split()))
    
    # For single-constraint knapsack, just one weight line
    weights = list(map(float, lines[2].split()))
    
    # Capacity
    capacity = float(lines[3])
    
    # Optimal (if available)
    optimal = None
    if len(lines) > 4:
        try:
            optimal = float(lines[4])
        except ValueError:
            pass
    
    return values, weights, capacity, optimal


def parse_tsplib_tour(filepath: str) -> List[int]:
    """
    Parse TSPLIB tour file (.tour).
    
    Args:
        filepath: Path to .tour file
        
    Returns:
        Tour as list of city indices
    """
    tour = []
    tour_section = False
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('TOUR_SECTION'):
                tour_section = True
                continue
            
            if tour_section:
                if line in ['-1', 'EOF']:
                    break
                
                try:
                    city = int(line)
                    tour.append(city - 1)  # Convert to 0-indexed
                except ValueError:
                    continue
    
    return tour


def detect_format(filepath: str) -> str:
    """
    Auto-detect file format.
    
    Args:
        filepath: Path to file
        
    Returns:
        Format string: 'tsplib', 'dimacs_graph', 'dimacs_cnf', 'knapsack'
    """
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
    
    if 'NAME' in first_line or 'DIMENSION' in first_line:
        return 'tsplib'
    elif first_line.startswith('p edge'):
        return 'dimacs_graph'
    elif first_line.startswith('p cnf'):
        return 'dimacs_cnf'
    else:
        return 'knapsack'


def download_sample_datasets(data_dir: str = 'data'):
    """
    Download small sample instances for testing.
    
    This would use requests to download from:
    - TSPLIB: att48, gr17, fri26
    - DIMACS: myciel3, queen5_5
    - SATLIB: uf20-91 (small satisfiable instances)
    
    Args:
        data_dir: Directory to save datasets
    """
    import requests
    from pathlib import Path
    
    base_path = Path(data_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Sample URLs (would need to be updated with actual links)
    samples = {
        'tsp/gr17.tsp': 'http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/gr17.tsp',
        # Add more samples here
    }
    
    for local_path, url in samples.items():
        dest = base_path / local_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            print(f"Downloading {url}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(dest, 'wb') as f:
                f.write(response.content)
            
            print(f"  Saved to {dest}")
        except Exception as e:
            print(f"  Failed: {e}")
