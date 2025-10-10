import math
import hashlib
import struct
from typing import TypeVar, Callable, List, Optional, Any, Set

T = TypeVar('T')

class BloomFilter:
    """
    Реалізація фільтра Блума (Bloom Filter).
    
    Фільтр Блума - імовірнісна структура даних, яка дозволяє ефективно перевіряти,
    чи належить елемент множині. Можливі помилкові спрацьовування (false positives),
    але не помилкові пропуски (false negatives).
    
    Time Complexity:
        - Додавання: O(k), де k - кількість хеш-функцій
        - Перевірка: O(k), де k - кількість хеш-функцій
    Space Complexity:
        - O(m), де m - розмір бітового масиву
    """
    
    def __init__(self, capacity: int, error_rate: float = 0.01):
        """
        Ініціалізує фільтр Блума.
        
        Args:
            capacity: очікувана максимальна кількість елементів
            error_rate: бажана ймовірність помилкових спрацьовувань
        """
        if capacity <= 0:
            raise ValueError("Ємність має бути більшою за нуль.")
        if error_rate <= 0 or error_rate >= 1:
            raise ValueError("Ймовірність помилки має бути в діапазоні (0, 1).")
        
        # Оптимальні параметри для фільтра Блума
        self.size = self._calculate_size(capacity, error_rate)
        self.hash_count = self._calculate_hash_count(self.size, capacity)
        
        # Бітовий масив (реалізовано як список булевих значень)
        self.bit_array = [False] * self.size
        
        # Лічильник доданих елементів
        self.count = 0
        
        # Зберігаємо параметри для статистики
        self.capacity = capacity
        self.error_rate = error_rate
    
    def _calculate_size(self, capacity: int, error_rate: float) -> int:
        """
        Обчислює оптимальний розмір бітового масиву.
        
        Formula: m = -n*ln(p)/(ln(2)^2), де:
            m - розмір бітового масиву
            n - ємність
            p - бажана ймовірність помилки
        """
        size = -capacity * math.log(error_rate) / (math.log(2) ** 2)
        return math.ceil(size)
    
    def _calculate_hash_count(self, size: int, capacity: int) -> int:
        """
        Обчислює оптимальну кількість хеш-функцій.
        
        Formula: k = (m/n)*ln(2), де:
            k - кількість хеш-функцій
            m - розмір бітового масиву
            n - ємність
        """
        hash_count = (size / capacity) * math.log(2)
        return math.ceil(hash_count)
    
    def _hash_functions(self, item: Any) -> List[int]:
        """
        Генерує k хеш-значень для елемента.
        
        Returns:
            Список індексів у бітовому масиві.
        """
        # Конвертуємо елемент в байти
        if isinstance(item, str):
            item_bytes = item.encode('utf-8')
        elif isinstance(item, (int, float)):
            item_bytes = str(item).encode('utf-8')
        elif isinstance(item, bytes):
            item_bytes = item
        else:
            item_bytes = str(item).encode('utf-8')
        
        # Обчислюємо два хеша для реалізації подвійного хешування
        h1 = int.from_bytes(hashlib.md5(item_bytes).digest(), byteorder='big')
        h2 = int.from_bytes(hashlib.sha1(item_bytes).digest(), byteorder='big')
        
        # Генеруємо k різних хешів за допомогою лінійної комбінації h1 і h2
        # (Метод Кірша-Міцемахера)
        return [(h1 + i * h2) % self.size for i in range(self.hash_count)]
    
    def add(self, item: Any) -> None:
        """
        Додає елемент до фільтра Блума.
        
        Args:
            item: елемент для додавання
        """
        for index in self._hash_functions(item):
            self.bit_array[index] = True
        self.count += 1
    
    def contains(self, item: Any) -> bool:
        """
        Перевіряє, чи міститься елемент у фільтрі.
        
        Args:
            item: елемент для перевірки
            
        Returns:
            True, якщо елемент, можливо, міститься у фільтрі (з деякою ймовірністю помилки).
            False, якщо елемент точно не міститься у фільтрі.
        """
        for index in self._hash_functions(item):
            if not self.bit_array[index]:
                return False
        return True
    
    def current_error_rate(self) -> float:
        """
        Обчислює поточну ймовірність помилкових спрацьовувань.
        
        Formula: (1 - e^(-k*n/m))^k, де:
            k - кількість хеш-функцій
            n - кількість доданих елементів
            m - розмір бітового масиву
            
        Returns:
            Поточна ймовірність помилкових спрацьовувань.
        """
        if self.count == 0:
            return 0.0
        
        # Обчислюємо наближення ймовірності
        rate = (1 - math.exp(-self.hash_count * self.count / self.size)) ** self.hash_count
        return min(1.0, rate)  # Обмежуємо значення до 1.0
    
    def stats(self) -> dict:
        """
        Повертає статистику фільтра Блума.
        
        Returns:
            Словник зі статистикою.
        """
        bit_array_usage = sum(self.bit_array) / self.size
        return {
            "size": self.size,
            "hash_functions": self.hash_count,
            "capacity": self.capacity,
            "count": self.count,
            "bit_array_usage": bit_array_usage,
            "target_error_rate": self.error_rate,
            "current_error_rate": self.current_error_rate()
        }

def main():
    """Демонстрація роботи фільтра Блума."""
    # Створення фільтра з ємністю 10,000 елементів і ймовірністю помилки 0.01
    bloom = BloomFilter(10000, 0.01)
    
    # Додаємо числа від 0 до 999
    for i in range(1000):
        bloom.add(i)
    
    # Перевіряємо числа від 0 до 1999
    present = 0
    false_positives = 0
    
    for i in range(2000):
        is_present = bloom.contains(i)
        if is_present and i < 1000:
            present += 1
        elif is_present and i >= 1000:
            false_positives += 1
    
    print("Фільтр Блума - демонстрація:")
    print(f"Додано елементів: {bloom.count}")
    print(f"Вірно виявлено: {present}/1000 (має бути 1000/1000)")
    print(f"Хибні спрацьовування: {false_positives}/1000")
    print(f"Фактична частота хибних спрацьовувань: {false_positives/1000:.4f}")
    print(f"Розрахункова частота хибних спрацьовувань: {bloom.current_error_rate():.4f}")
    
    # Виводимо статистику
    stats = bloom.stats()
    print("\nСтатистика фільтра Блума:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Порівняння з множиною
    import time
    import sys
    
    n = 1_000_000  # Кількість елементів
    elements = list(range(n))
    
    # Створення фільтра Блума і множини
    bloom_filter = BloomFilter(n, 0.01)
    set_collection = set()
    
    # Додавання елементів
    start_time = time.time()
    for element in elements:
        bloom_filter.add(element)
    bloom_time = time.time() - start_time
    
    start_time = time.time()
    for element in elements:
        set_collection.add(element)
    set_time = time.time() - start_time
    
    # Вимірювання розміру пам'яті (приблизно)
    bloom_memory = bloom_filter.size / 8  # в байтах (1 біт = 1/8 байта)
    set_memory = sys.getsizeof(set_collection)
    
    print("\nПорівняння з множиною для", n, "елементів:")
    print(f"Фільтр Блума: час={bloom_time:.4f}с, пам'ять={bloom_memory/1024/1024:.2f}МБ")
    print(f"Множина: час={set_time:.4f}с, пам'ять={set_memory/1024/1024:.2f}МБ")
    print(f"Економія пам'яті: {(1 - bloom_memory/set_memory) * 100:.2f}%")

if __name__ == "__main__":
    main()
