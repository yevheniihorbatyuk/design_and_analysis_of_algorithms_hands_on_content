# src/utils/parallel_utils_hybrid.py

import multiprocessing as mp
from collections import Counter
import time
from typing import Dict, List, Tuple
import os

# Встановлюємо оптимальні змінні середовища
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

def process_chunk_fast(data_chunk: List[str]) -> Dict[str, int]:
    """Оптимізована функція обробки з мінімальними витратами."""
    counts = Counter()
    # Використовуємо update замість ітерацій - швидше
    for line in data_chunk:
        counts.update(line.lower().split())
    return counts

def run_map_reduce_hybrid(data: List[str], num_workers: int = None) -> Tuple[dict, float]:
    """Гібридна версія з автоматичним підбором параметрів."""
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    start_time = time.perf_counter()
    
    # Оптимальний розмір chunk для мінімізації overhead
    total_items = len(data)
    optimal_chunk_size = max(1000, total_items // (num_workers * 4))
    chunks = [data[i:i + optimal_chunk_size] 
              for i in range(0, total_items, optimal_chunk_size)]
    
    # Використовуємо Pool з maxtasksperchild для уникнення витоків пам'яті
    with mp.Pool(processes=num_workers, maxtasksperchild=100) as pool:
        map_results = pool.map(process_chunk_fast, chunks, chunksize=1)
    
    # Швидке об'єднання через Counter.update
    final_counts = Counter()
    for result in map_results:
        final_counts.update(result)
    
    total_time = (time.perf_counter() - start_time) * 1000
    return dict(final_counts), total_time