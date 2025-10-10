import hashlib
import math
import struct
import random
from typing import Any, List, Dict, Tuple, Optional

class CountMinSketch:
    """
    Реалізація алгоритму Count-Min Sketch для приблизного підрахунку частоти елементів у потоці даних.
    
    Count-Min Sketch дозволяє оцінити частоту елементів, використовуючи набагато менше пам'яті,
    ніж повний словник. В обмін на це ми отримуємо приблизну оцінку, яка може мати помилку.
    
    Time Complexity:
        - Додавання: O(d), де d - кількість хеш-функцій
        - Оцінка частоти: O(d), де d - кількість хеш-функцій
    Space Complexity:
        - O(w * d), де w - ширина матриці, d - кількість хеш-функцій
    """
    
    def __init__(self, epsilon: float = 0.01, delta: float = 0.01):
        """
        Ініціалізує Count-Min Sketch з параметрами точності.
        
        Args:
            epsilon: параметр похибки (використовується для визначення ширини w)
            delta: параметр довіри (використовується для визначення глибини d)
        """
        # Обчислення оптимальних параметрів ширини (w) і глибини (d)
        self.w = math.ceil(math.e / epsilon)  # ширина
        self.d = math.ceil(math.log(1 / delta))  # глибина (кількість хеш-функцій)
        
        # Ініціалізація матриці лічильників
        self.table = [[0 for _ in range(self.w)] for _ in range(self.d)]
        
        # Ініціалізація випадкових параметрів для хеш-функцій
        # Будемо використовувати a*x + b mod p для хешування
        self.p = (1 << 31) - 1  # велике просте число (2^31 - 1)
        self.hash_params = [(random.randint(1, self.p - 1), random.randint(0, self.p - 1)) 
                           for _ in range(self.d)]
        
        # Лічильник доданих елементів
        self.total_count = 0
        
        # Зберігаємо параметри точності
        self.epsilon = epsilon
        self.delta = delta
    
    def _hash(self, item: Any) -> List[int]:
        """
        Хешує елемент d різними хеш-функціями.
        
        Args:
            item: елемент для хешування
            
        Returns:
            Список d хеш-значень (індексів у таблиці).
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
        
        # Створюємо базовий хеш за допомогою MD5
        md5_hash = hashlib.md5(item_bytes).digest()
        # Конвертуємо перші 4 байти в ціле число
        x = struct.unpack("<I", md5_hash[:4])[0]
        
        # Генеруємо d різних хеш-значень за допомогою лінійного конгруентного методу
        result = []
        for i in range(self.d):
            a, b = self.hash_params[i]
            hash_value = ((a * x + b) % self.p) % self.w
            result.append(hash_value)
        
        return result
    
    def add(self, item: Any, count: int = 1) -> None:
        """
        Додає елемент до Count-Min Sketch.
        
        Args:
            item: елемент для додавання
            count: кількість повторень (за замовчуванням 1)
        """
        if count <= 0:
            return
        
        hash_indices = self._hash(item)
        for i in range(self.d):
            self.table[i][hash_indices[i]] += count
        
        self.total_count += count
    
    def estimate_count(self, item: Any) -> int:
        """
        Оцінює частоту елемента.
        
        Args:
            item: елемент для оцінки
            
        Returns:
            Оцінка частоти елемента.
        """
        hash_indices = self._hash(item)
        # Повертаємо мінімальне значення серед всіх хеш-функцій
        return min(self.table[i][hash_indices[i]] for i in range(self.d))
    
    def estimate_relative_frequency(self, item: Any) -> float:
        """
        Оцінює відносну частоту елемента.
        
        Args:
            item: елемент для оцінки
            
        Returns:
            Оцінка відносної частоти елемента.
        """
        if self.total_count == 0:
            return 0.0
        return self.estimate_count(item) / self.total_count
    
    def heavy_hitters(self, threshold: float) -> List[Tuple[Any, int]]:
        """
        Знаходить "важкі елементи" (heavy hitters), частота яких перевищує поріг.
        
        Це приблизний метод, який потребує додаткового зберігання елементів.
        
        Args:
            threshold: поріг відносної частоти (0-1)
            
        Returns:
            Список кортежів (елемент, оцінка частоти).
        """
        # Це демонстраційний метод, який працює тільки якщо ми зберігаємо елементи окремо
        # У реальному використанні Count-Min Sketch цей метод не доступний без додаткового зберігання
        raise NotImplementedError(
            "Метод heavy_hitters не реалізований, оскільки Count-Min Sketch не зберігає самі елементи. "
            "Для знаходження важких елементів потрібно використовувати окремі структури даних, "
            "наприклад, алгоритм Space-Saving або структуру Count-Min-Heap."
        )
    
    def merge(self, other: 'CountMinSketch') -> 'CountMinSketch':
        """
        Об'єднує два Count-Min Sketch.
        
        Args:
            other: інший Count-Min Sketch (повинен мати ті самі параметри w і d)
            
        Returns:
            Новий Count-Min Sketch, що є об'єднанням двох.
        """
        if self.w != other.w or self.d != other.d:
            raise ValueError("Обидва Count-Min Sketch повинні мати однакові параметри w і d")
        
        result = CountMinSketch(self.epsilon, self.delta)
        
        # Складаємо відповідні комірки таблиць
        for i in range(self.d):
            for j in range(self.w):
                result.table[i][j] = self.table[i][j] + other.table[i][j]
        
        result.total_count = self.total_count + other.total_count
        
        return result
    
    def stats(self) -> Dict:
        """
        Повертає статистику Count-Min Sketch.
        
        Returns:
            Словник зі статистикою.
        """
        # Обчислюємо середнє, мінімум і максимум для комірок таблиці
        all_cells = [cell for row in self.table for cell in row]
        
        return {
            "width": self.w,
            "depth": self.d,
            "total_count": self.total_count,
            "memory_cells": self.w * self.d,
            "average_count": sum(all_cells) / (self.w * self.d) if all_cells else 0,
            "max_count": max(all_cells) if all_cells else 0,
            "min_count": min(all_cells) if all_cells else 0,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "error_bound": f"±{self.epsilon * self.total_count} з ймовірністю {1 - self.delta}"
        }

def main():
    """Демонстрація роботи алгоритму Count-Min Sketch."""
    import random
    import time
    import sys
    
    # Порівняння з точним підрахунком (словником)
    print("Count-Min Sketch - порівняння точності з прямим підрахунком:")
    
    # Параметри експерименту
    n = 1000000  # кількість елементів
    distinct = 10000  # кількість різних елементів
    zipf_param = 1.5  # параметр розподілу Zipf (більше значення = більша нерівномірність)
    
    # Генеруємо дані з розподілом Zipf (багато повторень найпопулярніших елементів)
    print(f"Генерація {n} елементів з розподілом Zipf (показник {zipf_param})...")
    
    # Визначення ваг для розподілу Zipf
    weights = [1.0 / (i ** zipf_param) for i in range(1, distinct + 1)]
    weight_sum = sum(weights)
    normalized_weights = [w / weight_sum for w in weights]
    
    # Генеруємо дані
    data = random.choices(range(distinct), weights=normalized_weights, k=n)
    
    # Прямий підрахунок (точний)
    exact_counts = {}
    
    start_time = time.time()
    for item in data:
        exact_counts[item] = exact_counts.get(item, 0) + 1
    exact_time = time.time() - start_time
    
    # Count-Min Sketch
    epsilon = 0.01  # похибка ±1%
    delta = 0.01  # довіра 99%
    
    cms = CountMinSketch(epsilon, delta)
    
    start_time = time.time()
    for item in data:
        cms.add(item)
    cms_time = time.time() - start_time
    
    # Порівняння точності
    print("\nПорівняння точності для топ-10 найчастіших елементів:")
    print("Елемент | Точне значення | Оцінка CMS | Відхилення")
    print("-" * 55)
    
    # Знаходимо топ-10 найчастіших елементів
    top_items = sorted(exact_counts.keys(), key=lambda x: exact_counts[x], reverse=True)[:10]
    
    total_error = 0
    for item in top_items:
        exact_count = exact_counts[item]
        cms_count = cms.estimate_count(item)
        error = cms_count - exact_count
        error_percent = error * 100 / exact_count if exact_count else 0
        total_error += abs(error)
        
        print(f"{item:7d} | {exact_count:14d} | {cms_count:10d} | {error:+d} ({error_percent:+.2f}%)")
    
    avg_error = total_error / len(top_items)
    
    # Порівняння швидкості та пам'яті
    exact_memory = sys.getsizeof(exact_counts)
    cms_stats = cms.stats()
    cms_memory = sys.getsizeof(cms.table)
    
    print(f"\nСереднє абсолютне відхилення для топ-10: {avg_error:.2f}")
    print(f"Очікувана похибка (ε × N): {epsilon * n:.2f}")
    
    print("\nПорівняння швидкості та пам'яті:")
    print(f"Точний підрахунок: час={exact_time:.4f}с, пам'ять={exact_memory/1024:.2f}КБ")
    print(f"Count-Min Sketch: час={cms_time:.4f}с, пам'ять={cms_memory/1024:.2f}КБ")
    print(f"Економія пам'яті: {(1 - cms_memory/exact_memory) * 100:.2f}%")
    
    # Демонстрація стійкості до помилок
    print("\nСтійкість до помилок (перевірка неіснуючих елементів):")
    
    false_positives = 0
    for i in range(distinct, distinct + 100):  # Перевіряємо елементи, яких немає в наборі
        if cms.estimate_count(i) > 0:
            false_positives += 1
    
    print(f"Частота помилкових спрацьовувань: {false_positives}%")
    
    # Статистика Count-Min Sketch
    print("\nСтатистика Count-Min Sketch:")
    for key, value in cms_stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
