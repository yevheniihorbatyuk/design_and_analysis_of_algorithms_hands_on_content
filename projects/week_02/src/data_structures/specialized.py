# src/data_structures/specialized.py

import heapq
from collections import OrderedDict
import hashlib

class PriorityQueue:
    """Проста обгортка над heapq для наочності."""
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        # heapq — це min-heap, тому для max-heap можна використовувати негативні пріоритети
        heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]
        
    def is_empty(self):
        return len(self._queue) == 0



class BloomFilter:
    """
    Ймовірнісна структура даних для перевірки належності до множини.
    Хибно-негативні неможливі; хибно-позитивні можливі.
    Реалізація: бітове поле на bytearray + double hashing (Kirsch–Mitzenmacher).
    """
    def __init__(self, size: int, hash_count: int):
        self.size = int(size)
        self.hash_count = int(hash_count)
        self.bits = bytearray((self.size + 7) // 8)

    # ---- бітові операції ----
    def _setbit(self, idx: int) -> None:
        self.bits[idx >> 3] |= (1 << (idx & 7))

    def _getbit(self, idx: int) -> int:
        return (self.bits[idx >> 3] >> (idx & 7)) & 1

    # ---- double hashing: k індексів з двох базових хешів ----
    def _hashes(self, item):
        s = str(item).encode("utf-8")
        # Швидша за sha256: беремо blake2b і ділимо на 2 частини
        d = hashlib.blake2b(s, digest_size=16).digest()
        h1 = int.from_bytes(d[:8], "big")
        h2 = int.from_bytes(d[8:], "big") or 0x9E3779B97F4A7C15  # якщо 0, беремо фіксований крок
        m = self.size
        for i in range(self.hash_count):
            yield (h1 + i * h2) % m

    def add(self, item) -> None:
        for idx in self._hashes(item):
            self._setbit(idx)

    def __contains__(self, item) -> bool:
        for idx in self._hashes(item):
            if not self._getbit(idx):
                return False
        return True

    # Оцінка FPR для n вставлених елементів
    @staticmethod
    def estimate_fpr(m: int, k: int, n: int) -> float:
        # (1 - e^{-kn/m})^k
        if m <= 0: return 1.0
        return (1.0 - math.exp(-k * n / m)) ** k


class LRUCache:
    """
    Кеш з політикою витіснення "Least Recently Used" (той, що найдовше не використовувався).
    Реалізований за допомогою OrderedDict для простоти.
    """
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: int):
        if key not in self.cache:
            return -1
        # Переміщуємо елемент в кінець, щоб позначити його як "недавно використаний"
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int):
        if key in self.cache:
            # Оновлюємо і переміщуємо в кінець
            self.cache[key] = value
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                # Видаляємо перший (найстаріший) елемент
                self.cache.popitem(last=False)
            self.cache[key] = value