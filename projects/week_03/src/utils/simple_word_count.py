# src/utils/simple_word_count.py

from collections import defaultdict, Counter
import time
from typing import Dict, List, Tuple

# ============================================================================
# Прості версії БЕЗ паралелізму для baseline порівняння
# ============================================================================

def simple_word_count_defaultdict(data: List[str]) -> Tuple[Dict[str, int], float]:
    """Найпростіша версія з defaultdict."""
    start_time = time.perf_counter()
    
    counts = defaultdict(int)
    for line in data:
        for word in line.strip().lower().split():
            counts[word] += 1
    
    total_time = (time.perf_counter() - start_time) * 1000
    return dict(counts), total_time


def simple_word_count_counter(data: List[str]) -> Tuple[Dict[str, int], float]:
    """Оптимізована проста версія з Counter."""
    start_time = time.perf_counter()
    
    counts = Counter()
    for line in data:
        counts.update(line.lower().split())
    
    total_time = (time.perf_counter() - start_time) * 1000
    return dict(counts), total_time


def simple_word_count_dict(data: List[str]) -> Tuple[Dict[str, int], float]:
    """Версія з чистим dict (найшвидша послідовна)."""
    start_time = time.perf_counter()
    
    counts = {}
    for line in data:
        for word in line.lower().split():
            counts[word] = counts.get(word, 0) + 1
    
    total_time = (time.perf_counter() - start_time) * 1000
    return counts, total_time


def simple_word_count_optimized(data: List[str]) -> Tuple[Dict[str, int], float]:
    """Максимально оптимізована послідовна версія."""
    start_time = time.perf_counter()
    
    counts = {}
    get_count = counts.get  # Кешуємо метод
    
    for line in data:
        words = line.lower().split()
        for word in words:
            counts[word] = get_count(word, 0) + 1
    
    total_time = (time.perf_counter() - start_time) * 1000
    return counts, total_time