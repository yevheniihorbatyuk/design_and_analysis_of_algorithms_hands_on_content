# src/utils/parallel_utils_optimized.py

import multiprocessing
from collections import Counter
from typing import Dict, List, Tuple, DefaultDict
import time
import numpy as np


def map_reduce_word_count(data_chunk: List[str]) -> DefaultDict:
    """Map-фаза: рахує слова у фрагменті."""
    counts = DefaultDict(int)
    for line in data_chunk:
        for word in line.strip().lower().split():
            counts[word] += 1
    return counts

def run_map_reduce(data: List[str], num_workers: int) -> Tuple[dict, float]:
    """Виконує повний цикл MapReduce."""
    start_time = time.perf_counter()

    # Розбиття даних на частини
    chunk_size = int(np.ceil(len(data) / num_workers))
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    # Map-фаза паралельно
    with multiprocessing.Pool(num_workers) as pool:
        map_results = pool.map(map_reduce_word_count, chunks)

    # Reduce-фаза (shuffle + reduce)
    final_counts = DefaultDict(int)
    for res in map_results:
        for word, count in res.items():
            final_counts[word] += count

    total_time = (time.perf_counter() - start_time) * 1000
    return dict(final_counts), total_time
