# src/data_structures/probabilistic.py

import math
import mmh3
from bitarray import bitarray
from collections import defaultdict, deque
import random
import numpy as np
from typing import List, Iterator, TypeVar, Generic, Any

T = TypeVar('T')

class BloomFilter:
    """Реалізація фільтра Блума."""
    def __init__(self, size: int, hash_count: int):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)

    @classmethod
    def from_capacity(cls, capacity: int, error_rate: float = 0.01):
        size = int(-(capacity * math.log(error_rate)) / (math.log(2) ** 2))
        hash_count = int((size / capacity) * math.log(2))
        return cls(size, max(1, hash_count))

    def add(self, item: str):
        for i in range(self.hash_count):
            index = mmh3.hash(item, i) % self.size
            self.bit_array[index] = 1

    def __contains__(self, item: str) -> bool:
        for i in range(self.hash_count):
            index = mmh3.hash(item, i) % self.size
            if not self.bit_array[index]:
                return False
        return True

class HyperLogLog:
    """Реалізація HyperLogLog."""
    def __init__(self, precision: int = 14):
        self.p = precision
        self.m = 1 << precision
        self.registers = [0] * self.m
        self.alpha = self._get_alpha()

    def _get_alpha(self) -> float:
        if self.m <= 16: return 0.673
        if self.m == 32: return 0.697
        if self.m == 64: return 0.709
        return 0.7213 / (1 + 1.079 / self.m)

    def add(self, item: str):
        x = mmh3.hash128(item, signed=False)
        j = x & (self.m - 1)
        w = x >> self.p
        leading_zeros = (128 - self.p - w.bit_length() + 1) if w > 0 else 128 - self.p
        self.registers[j] = max(self.registers[j], leading_zeros)

    def estimate(self) -> float:
        sum_inv = sum(math.pow(2.0, -reg) for reg in self.registers)
        estimate = self.alpha * self.m * self.m / sum_inv
        zeros = self.registers.count(0)
        if estimate <= 2.5 * self.m and zeros > 0:
            return self.m * math.log(self.m / zeros)
        return estimate

class CountMinSketch:
    """Реалізація Count-Min Sketch."""
    def __init__(self, width: int, depth: int):
        self.width = width
        self.depth = depth
        self.table = [[0] * width for _ in range(depth)]

    def add(self, item: str):
        for i in range(self.depth):
            index = mmh3.hash(item, i) % self.width
            self.table[i][index] += 1

    def estimate(self, item: str) -> int:
        return min(self.table[i][mmh3.hash(item, i) % self.width] for i in range(self.depth))

class ReservoirSampling(Generic[T]):
    """Реалізація резервуарної вибірки."""
    def __init__(self, k: int):
        self.k = k
        self.reservoir: List[T] = []
        self.items_seen = 0

    def process_stream(self, stream: Iterator[T]):
        for item in stream:
            self.items_seen += 1
            if len(self.reservoir) < self.k:
                self.reservoir.append(item)
            else:
                j = random.randint(0, self.items_seen - 1)
                if j < self.k:
                    self.reservoir[j] = item
    
    def get_sample(self) -> List[T]:
        return self.reservoir

class MisraGries(Generic[T]):
    """Реалізація алгоритму Misra-Gries."""
    def __init__(self, k: int):
        self.k = k
        self.counters = defaultdict(int)

    def process_stream(self, stream: Iterator[T]):
        for item in stream:
            if item in self.counters:
                self.counters[item] += 1
            elif len(self.counters) < self.k - 1:
                self.counters[item] = 1
            else:
                for key in list(self.counters.keys()):
                    self.counters[key] -= 1
                    if self.counters[key] == 0:
                        del self.counters[key]
    
    def get_frequent_items(self) -> dict:
        return dict(self.counters)

class MinHashLSH:
    """Реалізація LSH на основі MinHash для подібності Жаккара."""
    def __init__(self, num_hashes=128, threshold=0.5):
        self.num_hashes = num_hashes
        self.threshold = threshold
        # Генеруємо параметри для хеш-функцій
        self.hash_params = [(random.randint(1, 2**32-2), random.randint(0, 2**32-2)) for _ in range(num_hashes)]
        self.bands = int(num_hashes / 2) # Приклад: ділимо на смуги по 2 хеші
        self.rows = 2
        self.buckets = [defaultdict(list) for _ in range(self.bands)]

    def _minhash(self, item_set: set):
        signature = []
        for a, b in self.hash_params:
            min_hash = float('inf')
            for item in item_set:
                hash_val = (a * mmh3.hash(str(item)) + b) % (2**32-1)
                if hash_val < min_hash:
                    min_hash = hash_val
            signature.append(min_hash)
        return signature

    def add(self, item_id: Any, item_set: set):
        signature = self._minhash(item_set)
        for i in range(self.bands):
            band = signature[i * self.rows : (i + 1) * self.rows]
            bucket_key = hash(tuple(band))
            self.buckets[i][bucket_key].append(item_id)

    def query(self, item_set: set) -> set:
        signature = self._minhash(item_set)
        candidates = set()
        for i in range(self.bands):
            band = signature[i * self.rows : (i + 1) * self.rows]
            bucket_key = hash(tuple(band))
            if bucket_key in self.buckets[i]:
                for candidate in self.buckets[i][bucket_key]:
                    candidates.add(candidate)
        return candidates

class SlidingWindow:
    """Обчислення агрегацій у ковзному вікні."""
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.window = deque(maxlen=window_size)
        self.sum = 0.0

    def add(self, value: float):
        if len(self.window) == self.window_size:
            self.sum -= self.window[0]
        self.window.append(value)
        self.sum += value

    def get_average(self) -> float:
        if not self.window:
            return 0.0
        return self.sum / len(self.window)