# src/utils/parallel_utils_optimized.py

import multiprocessing
from collections import Counter
from typing import Dict, List, Tuple, DefaultDict
import time
import numpy as np
from numba import njit, typed, types
from numba.typed import Dict as NumbaDict

# ============================================================================
# Оптимізація 1: Numba JIT для підрахунку слів
# ============================================================================

@njit
def count_words_numba(words_array, word_to_idx, counts):
    """Швидкий підрахунок слів через Numba."""
    for word_idx in words_array:
        counts[word_idx] += 1
    return counts

def map_reduce_word_count_optimized(data_chunk: List[str]) -> Dict[str, int]:
    """Map-фаза з використанням Counter (швидше ніж defaultdict)."""
    counts = Counter()
    for line in data_chunk:
        # Оптимізація: split() без strip() якщо дані чисті
        counts.update(line.lower().split())
    return counts

# ============================================================================
# Оптимізація 2: Batch processing з numpy
# ============================================================================

def map_reduce_batch(data_chunk: List[str]) -> Dict[str, int]:
    """Обробка батчами для кращої локальності кешу."""
    counts = Counter()
    batch_size = 1000
    
    for i in range(0, len(data_chunk), batch_size):
        batch = data_chunk[i:i + batch_size]
        # Обробка батчу
        for line in batch:
            counts.update(line.lower().split())
    
    return counts

# ============================================================================
# Оптимізація 3: Паралельна reduce фаза
# ============================================================================

def parallel_reduce(map_results: List[Dict], num_workers: int) -> Dict[str, int]:
    """Паралельна reduce-фаза."""
    if len(map_results) <= 1:
        return map_results[0] if map_results else {}
    
    # Групуємо результати для паралельного злиття
    def merge_dicts(dicts_pair):
        result = Counter()
        for d in dicts_pair:
            result.update(d)
        return result
    
    # Рекурсивне злиття парами
    while len(map_results) > 1:
        pairs = []
        for i in range(0, len(map_results), 2):
            if i + 1 < len(map_results):
                pairs.append([map_results[i], map_results[i + 1]])
            else:
                pairs.append([map_results[i]])
        
        with multiprocessing.Pool(min(num_workers, len(pairs))) as pool:
            map_results = pool.map(merge_dicts, pairs)
    
    return dict(map_results[0])

# ============================================================================
# Основна функція з оптимізаціями
# ============================================================================

def run_map_reduce_optimized(data: List[str], num_workers: int) -> Tuple[dict, float]:
    """Оптимізований MapReduce."""
    start_time = time.perf_counter()
    
    # Оптимізація розбиття: вирівнювання для кращого балансування
    chunk_size = max(1, len(data) // num_workers)
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    # Map-фаза з chunksize для зменшення overhead
    with multiprocessing.Pool(num_workers) as pool:
        map_results = pool.map(map_reduce_word_count_optimized, chunks, chunksize=1)
    
    # Швидка reduce-фаза через Counter
    final_counts = Counter()
    for res in map_results:
        final_counts.update(res)
    
    total_time = (time.perf_counter() - start_time) * 1000
    return dict(final_counts), total_time

# ============================================================================
# ULTRA-FAST версія з мінімальним overhead
# ============================================================================

def map_reduce_fast(data_chunk: List[str]) -> Dict[str, int]:
    """Максимально швидка версія без зайвих операцій."""
    counts = {}
    for line in data_chunk:
        for word in line.lower().split():
            counts[word] = counts.get(word, 0) + 1
    return counts

def run_map_reduce_ultra_fast(data: List[str], num_workers: int) -> Tuple[dict, float]:
    """ULTRA швидка версія з мінімальними витратами."""
    start_time = time.perf_counter()
    
    # Розбиття без numpy
    chunk_size = len(data) // num_workers + (1 if len(data) % num_workers else 0)
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    # Map phase
    with multiprocessing.Pool(num_workers) as pool:
        map_results = pool.map(map_reduce_fast, chunks, chunksize=1)
    
    # Reduce phase - найшвидший варіант
    final_counts = {}
    for res in map_results:
        for word, count in res.items():
            final_counts[word] = final_counts.get(word, 0) + count
    
    total_time = (time.perf_counter() - start_time) * 1000
    return final_counts, total_time