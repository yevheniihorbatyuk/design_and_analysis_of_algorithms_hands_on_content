import math
import hashlib
import struct
from typing import Any, List, Dict

class HyperLogLog:
    """
    Реалізація алгоритму HyperLogLog для приблизного підрахунку кількості унікальних елементів.
    
    HyperLogLog - це алгоритм, що дозволяє оцінити потужність мультимножини
    з високою точністю, використовуючи малу кількість пам'яті.
    
    Time Complexity:
        - Додавання: O(1)
        - Оцінка потужності: O(m), де m - кількість регістрів
    Space Complexity:
        - O(2^p), де p - кількість біт для адресації регістрів
    """
    
    def __init__(self, p: int = 10):
        """
        Ініціалізує HyperLogLog.
        
        Args:
            p: кількість біт, що використовуються для адресації регістрів (4 <= p <= 16).
               Визначає розмір масиву регістрів (m = 2^p).
        """
        if p < 4 or p > 16:
            raise ValueError("Параметр p має бути в діапазоні [4, 16]")
        
        self.p = p
        self.m = 1 << p  # m = 2^p
        self.registers = [0] * self.m
        
        # Constants for bias correction
        self.alpha = self._get_alpha(self.m)
    
    def _get_alpha(self, m: int) -> float:
        """
        Повертає поправочний коефіцієнт alpha для формули оцінки потужності.
        
        Args:
            m: кількість регістрів
            
        Returns:
            Значення коефіцієнта alpha.
        """
        if m == 16:
            return 0.673
        elif m == 32:
            return 0.697
        elif m == 64:
            return 0.709
        else:
            return 0.7213 / (1.0 + 1.079 / m)
    
    def _hash(self, item: Any) -> int:
        """
        Обчислює хеш елемента.
        
        Args:
            item: елемент для хешування
            
        Returns:
            32-бітний хеш.
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
        
        # Використовуємо MD5 для отримання хешу
        hash_value = hashlib.md5(item_bytes).digest()
        # Перетворюємо перші 4 байти в ціле число
        return struct.unpack("<I", hash_value[:4])[0]
    
    def _get_register_index(self, hash_value: int) -> int:
        """
        Отримує індекс регістра з хешу.
        
        Args:
            hash_value: хеш елемента
            
        Returns:
            Індекс регістра (перші p біт хешу).
        """
        # Використовуємо перші p біт хешу як індекс регістра
        return hash_value & (self.m - 1)
    
    def _get_leading_zeros(self, hash_value: int) -> int:
        """
        Обчислює кількість провідних нулів у двійковому представленні хешу.
        
        Args:
            hash_value: хеш елемента
            
        Returns:
            Кількість провідних нулів + 1.
        """
        # Скидаємо перші p біт, які ми вже використали для індексу
        pattern = hash_value >> self.p
        
        # Знаходимо позицію першої 1 (рахуючи з 1)
        rank = 1
        while pattern & 1 == 0 and rank <= 32 - self.p:
            pattern >>= 1
            rank += 1
        
        return rank
    
    def add(self, item: Any) -> None:
        """
        Додає елемент до HyperLogLog.
        
        Args:
            item: елемент для додавання
        """
        hash_value = self._hash(item)
        register_index = self._get_register_index(hash_value)
        leading_zeros = self._get_leading_zeros(hash_value)
        
        # Оновлюємо регістр максимальним значенням
        self.registers[register_index] = max(self.registers[register_index], leading_zeros)
    
    def count(self) -> int:
        """
        Оцінює кількість унікальних елементів.
        
        Returns:
            Оцінка кількості унікальних елементів.
        """
        # Обчислюємо гармонічне середнє
        sum_inverses = sum(math.pow(2, -register) for register in self.registers)
        raw_estimate = self.alpha * (self.m ** 2) / sum_inverses
        
        # Застосовуємо корекцію для малих і великих значень
        if raw_estimate <= 2.5 * self.m:  # Small range correction
            # Підраховуємо кількість регістрів зі значенням 0
            zeros = self.registers.count(0)
            if zeros > 0:
                return int(self.m * math.log(self.m / zeros))
            else:
                return int(raw_estimate)
        elif raw_estimate <= (1.0 / 30.0) * (1 << 32):  # No correction needed
            return int(raw_estimate)
        else:  # Large range correction
            return int(-(1 << 32) * math.log(1 - raw_estimate / (1 << 32)))
    
    def merge(self, other: 'HyperLogLog') -> 'HyperLogLog':
        """
        Об'єднує два HyperLogLog.
        
        Args:
            other: інший HyperLogLog (повинен мати той самий параметр p)
            
        Returns:
            Новий HyperLogLog, що є об'єднанням двох.
        """
        if self.p != other.p:
            raise ValueError("Обидва HyperLogLog повинні мати однаковий параметр p")
        
        result = HyperLogLog(self.p)
        result.registers = [max(a, b) for a, b in zip(self.registers, other.registers)]
        return result
    
    def stats(self) -> Dict:
        """
        Повертає статистику HyperLogLog.
        
        Returns:
            Словник зі статистикою.
        """
        return {
            "p": self.p,
            "m": self.m,
            "count": self.count(),
            "registers_mean": sum(self.registers) / self.m,
            "registers_max": max(self.registers),
            "registers_min": min(self.registers),
            "zero_registers": self.registers.count(0),
            "memory_bytes": self.m,  # Кожен регістр займає 1 байт
            "standard_error": 1.04 / math.sqrt(self.m)  # Стандартна похибка ~1.04/sqrt(m)
        }

def main():
    """Демонстрація роботи алгоритму HyperLogLog."""
    import random
    import time
    import sys
    
    # Порівняння точності для різних значень p
    p_values = [6, 8, 10, 12, 14]
    true_cardinality = 100000
    
    print("HyperLogLog - демонстрація точності для різних значень p:")
    print(f"Істинна кількість унікальних елементів: {true_cardinality}")
    print("\np\tm\tОцінка\tПохибка\tПам'ять")
    
    for p in p_values:
        hll = HyperLogLog(p)
        for i in range(true_cardinality):
            hll.add(i)
        
        estimate = hll.count()
        error_percent = abs(estimate - true_cardinality) / true_cardinality * 100
        memory_bytes = hll.m
        
        print(f"{p}\t{hll.m}\t{estimate}\t{error_percent:.2f}%\t{memory_bytes} байт")
    
    # Порівняння з множиною (Set)
    print("\nПорівняння HyperLogLog з множиною Python:")
    
    cardinalities = [10000, 100000, 1000000, 10000000]
    
    for cardinality in cardinalities:
        # Створення даних
        data = list(range(cardinality))
        
        # HyperLogLog
        p = 12  # Гарне значення для більшості застосувань
        hll = HyperLogLog(p)
        
        start_time = time.time()
        for item in data:
            hll.add(item)
        hll_time = time.time() - start_time
        
        hll_estimate = hll.count()
        hll_error = abs(hll_estimate - cardinality) / cardinality * 100
        hll_memory = hll.m
        
        # Множина Python
        py_set = set()
        
        start_time = time.time()
        for item in data:
            py_set.add(item)
        set_time = time.time() - start_time
        
        set_count = len(py_set)
        set_memory = sys.getsizeof(py_set)
        
        print(f"\nКількість елементів: {cardinality}")
        print(f"HyperLogLog: оцінка={hll_estimate}, похибка={hll_error:.2f}%, час={hll_time:.4f}с, пам'ять={hll_memory} байт")
        print(f"Множина Python: count={set_count}, похибка=0.00%, час={set_time:.4f}с, пам'ять={set_memory} байт")
        print(f"Економія пам'яті: {(1 - hll_memory/set_memory) * 100:.2f}%")

    # Тестування на збіжність
    print("\nТестування збіжності HyperLogLog:")
    
    cardinality = 1000000
    iterations = 10
    
    hll = HyperLogLog(12)
    
    print("Ітерація\tДодано елементів\tОцінка\tПохибка")
    for i in range(iterations):
        batch_size = cardinality // iterations
        for j in range(batch_size):
            hll.add(i * batch_size + j)
        
        current_elements = (i + 1) * batch_size
        estimate = hll.count()
        error = abs(estimate - current_elements) / current_elements * 100
        
        print(f"{i+1}\t{current_elements}\t{estimate}\t{error:.2f}%")
    
    # Тестування об'єднання HyperLogLog
    print("\nТестування об'єднання HyperLogLog:")
    
    hll1 = HyperLogLog(12)
    hll2 = HyperLogLog(12)
    
    # Додаємо різні елементи в кожен HyperLogLog
    for i in range(100000):
        hll1.add(f"set1-{i}")
    
    for i in range(50000):
        hll2.add(f"set2-{i}")
    
    # Додаємо перетин
    for i in range(25000):
        hll1.add(f"common-{i}")
        hll2.add(f"common-{i}")
    
    # Об'єднуємо та порівнюємо результати
    hll_merged = hll1.merge(hll2)
    
    print(f"HLL1 оцінка: {hll1.count()}")
    print(f"HLL2 оцінка: {hll2.count()}")
    print(f"Об'єднання оцінка: {hll_merged.count()}")
    print(f"Очікувана кількість унікальних елементів: {100000 + 50000 + 25000 - 25000}")

if __name__ == "__main__":
    main()
