import mmh3  # Пакет для хешування
from bitarray import bitarray  # Ефективний бітовий масив


class BloomFilter:
    """
    Реалізація фільтра Блума для ефективної перевірки членства.
    """
    def __init__(self, size: int, hash_count: int):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)
    
    @classmethod
    def from_capacity(cls, expected_elements: int, false_positive_rate: float) -> 'BloomFilter':
        """
        Створює фільтр Блума з оптимальними параметрами.
        
        Args:
            expected_elements: Очікувана кількість елементів
            false_positive_rate: Допустима ймовірність помилки
            
        Returns:
            BloomFilter: Оптимально налаштований фільтр
        """
        size = cls.calculate_size(expected_elements, false_positive_rate)
        hash_count = cls.calculate_hash_count(size, expected_elements)
        return cls(size, hash_count)
    
    @staticmethod
    def calculate_size(n: int, p: float) -> int:
        """Розраховує оптимальний розмір бітового масиву."""
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(m)
    
    @staticmethod
    def calculate_hash_count(m: int, n: int) -> int:
        """Розраховує оптимальну кількість хеш-функцій."""
        k = (m / n) * math.log(2)
        return int(k)
    
    def add(self, item: str) -> None:
        """Додає елемент до фільтра."""
        for seed in range(self.hash_count):
            index = mmh3.hash(str(item), seed) % self.size
            self.bit_array[index] = 1
    
    def check(self, item: str) -> bool:
        """Перевіряє чи елемент можливо присутній у фільтрі."""
        for seed in range(self.hash_count):
            index = mmh3.hash(str(item), seed) % self.size
            if not self.bit_array[index]:
                return False
        return True



if __name__=='__main__':
    # Тестування
    bloom = BloomFilter(1000, 5)
    bloom.add("apple")
    bloom.add("banana")

    print("apple in filter?", bloom.check("apple"))  # True
    print("grape in filter?", bloom.check("grape"))  # False (можливо True через false positive)