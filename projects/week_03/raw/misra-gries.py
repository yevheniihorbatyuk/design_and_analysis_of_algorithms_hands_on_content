from typing import Dict, List, Any, TypeVar, Tuple, Set, Optional, Generic
from collections import defaultdict
import heapq

T = TypeVar('T')

class MisraGries(Generic[T]):
    """
    Реалізація алгоритму Misra-Gries для визначення частих елементів у потоці даних.
    
    Цей алгоритм дозволяє знаходити елементи, частота яких перевищує заданий поріг,
    використовуючи обмежену кількість пам'яті. Він гарантує знаходження всіх елементів,
    частота яких перевищує N/k, де N - загальна кількість елементів, а k - параметр алгоритму.
    
    Time Complexity:
        - Додавання: O(1) амортизовано
        - Отримання частих елементів: O(k)
    Space Complexity:
        - O(k), де k - кількість лічильників
    """
    
    def __init__(self, k: int):
        """
        Ініціалізує алгоритм Misra-Gries.
        
        Args:
            k: кількість лічильників (параметр точності)
        """
        if k <= 0:
            raise ValueError("Кількість лічильників повинна бути більше 0")
        
        self.k = k  # Кількість лічильників
        self.counters: Dict[T, int] = {}  # Лічильники для елементів
        self.total_count = 0  # Загальна кількість елементів
    
    def add(self, item: T) -> None:
        """
        Додає елемент до потоку.
        
        Args:
            item: елемент для додавання
        """
        self.total_count += 1
        
        # Якщо елемент вже в лічильниках, збільшуємо його кількість
        if item in self.counters:
            self.counters[item] += 1
        # Якщо є вільне місце, додаємо елемент
        elif len(self.counters) < self.k:
            self.counters[item] = 1
        # Інакше зменшуємо всі лічильники на 1
        else:
            to_remove = []
            for key in self.counters:
                self.counters[key] -= 1
                if self.counters[key] == 0:
                    to_remove.append(key)
            
            # Видаляємо елементи з нульовими лічильниками
            for key in to_remove:
                del self.counters[key]
    
    def process_batch(self, items: List[T]) -> None:
        """
        Обробляє пакет елементів.
        
        Args:
            items: список елементів для обробки
        """
        for item in items:
            self.add(item)
    
    def get_frequent_items(self, threshold: float = None) -> Dict[T, int]:
        """
        Повертає часті елементи.
        
        Args:
            threshold: поріг відносної частоти (0-1).
                      Якщо None, використовується 1/k.
        
        Returns:
            Словник {елемент: кількість} для частих елементів.
        """
        if threshold is None:
            threshold = 1.0 / self.k
        
        threshold_count = threshold * self.total_count
        
        return {item: count for item, count in self.counters.items() 
                if count > threshold_count}
    
    def get_top_k(self, k: Optional[int] = None) -> List[Tuple[T, int]]:
        """
        Повертає k найчастіших елементів.
        
        Args:
            k: кількість елементів для повернення.
               Якщо None, повертає всі елементи в лічильниках.
        
        Returns:
            Список кортежів (елемент, кількість) для k найчастіших елементів.
        """
        if k is None:
            k = len(self.counters)
        
        return sorted(self.counters.items(), key=lambda x: x[1], reverse=True)[:k]
    
    def merge(self, other: 'MisraGries[T]') -> 'MisraGries[T]':
        """
        Об'єднує два екземпляри Misra-Gries.
        
        Зауваження: це приблизне об'єднання, яке не гарантує повної точності.
        
        Args:
            other: інший екземпляр Misra-Gries
        
        Returns:
            Новий екземпляр Misra-Gries, що є об'єднанням двох.
        """
        # Визначаємо параметр k для результату (мінімум з двох)
        k = min(self.k, other.k)
        result = MisraGries(k)
        
        # Об'єднуємо лічильники
        all_items = set(self.counters.keys()) | set(other.counters.keys())
        for item in all_items:
            count = self.counters.get(item, 0) + other.counters.get(item, 0)
            if count > 0:
                result.counters[item] = count
        
        # Якщо об'єднаних лічильників більше k, видаляємо найменші
        if len(result.counters) > k:
            items_by_count = sorted(result.counters.items(), key=lambda x: x[1])
            for item, _ in items_by_count[:len(result.counters) - k]:
                del result.counters[item]
        
        result.total_count = self.total_count + other.total_count
        
        return result
    
    def stats(self) -> Dict:
        """
        Повертає статистику алгоритму Misra-Gries.
        
        Returns:
            Словник зі статистикою.
        """
        return {
            "k": self.k,
            "active_counters": len(self.counters),
            "total_count": self.total_count,
            "min_frequency_threshold": 1.0 / self.k,
            "theoretical_error_bound": self.total_count / self.k
        }


class SpaceSaving(Generic[T]):
    """
    Реалізація алгоритму Space-Saving - покращеної версії алгоритму Misra-Gries.
    
    Space-Saving забезпечує більш точні оцінки частоти елементів порівняно з Misra-Gries.
    Він також гарантує знаходження всіх елементів, частота яких перевищує N/k.
    
    Time Complexity:
        - Додавання: O(1) амортизовано
        - Отримання частих елементів: O(k log k)
    Space Complexity:
        - O(k), де k - кількість лічильників
    """
    
    def __init__(self, k: int):
        """
        Ініціалізує алгоритм Space-Saving.
        
        Args:
            k: кількість лічильників (параметр точності)
        """
        if k <= 0:
            raise ValueError("Кількість лічильників повинна бути більше 0")
        
        self.k = k  # Кількість лічильників
        self.counters: Dict[T, int] = {}  # Лічильники для елементів
        self.errors: Dict[T, int] = {}  # Максимальна похибка для кожного елемента
        self.total_count = 0  # Загальна кількість елементів
    
    def add(self, item: T) -> None:
        """
        Додає елемент до потоку.
        
        Args:
            item: елемент для додавання
        """
        self.total_count += 1
        
        # Якщо елемент вже в лічильниках, збільшуємо його кількість
        if item in self.counters:
            self.counters[item] += 1
        # Якщо є вільне місце, додаємо елемент
        elif len(self.counters) < self.k:
            self.counters[item] = 1
            self.errors[item] = 0
        # Інакше замінюємо елемент з найменшою частотою
        else:
            # Знаходимо елемент з найменшою частотою
            min_item = min(self.counters, key=self.counters.get)
            min_count = self.counters[min_item]
            
            # Запам'ятовуємо похибку для нового елемента
            error = min_count
            
            # Видаляємо елемент з найменшою частотою
            del self.counters[min_item]
            del self.errors[min_item]
            
            # Додаємо новий елемент з лічильником, збільшеним на 1
            self.counters[item] = min_count + 1
            self.errors[item] = error
    
    def get_count(self, item: T) -> Tuple[int, int]:
        """
        Повертає оцінку частоти елемента та максимальну похибку.
        
        Args:
            item: елемент для оцінки
        
        Returns:
            Кортеж (оцінка частоти, максимальна похибка).
        """
        if item in self.counters:
            return self.counters[item], self.errors[item]
        else:
            # Якщо елемент не відстежується, найбільша можлива похибка -
            # це найменша частота відстежуваних елементів
            min_count = min(self.counters.values()) if self.counters else 0
            return 0, min_count
    
    def get_top_k(self, k: Optional[int] = None) -> List[Tuple[T, int, int]]:
        """
        Повертає k найчастіших елементів.
        
        Args:
            k: кількість елементів для повернення.
               Якщо None, повертає всі елементи в лічильниках.
        
        Returns:
            Список кортежів (елемент, оцінка частоти, максимальна похибка).
        """
        if k is None:
            k = len(self.counters)
        
        items = [(item, count, self.errors[item]) 
                for item, count in self.counters.items()]
        return sorted(items, key=lambda x: x[1], reverse=True)[:k]
    
    def stats(self) -> Dict:
        """
        Повертає статистику алгоритму Space-Saving.
        
        Returns:
            Словник зі статистикою.
        """
        return {
            "k": self.k,
            "active_counters": len(self.counters),
            "total_count": self.total_count,
            "min_frequency_threshold": 1.0 / self.k,
            "theoretical_error_bound": self.total_count / self.k,
            "max_error": max(self.errors.values()) if self.errors else 0,
            "avg_error": sum(self.errors.values()) / len(self.errors) if self.errors else 0
        }


def main():
    """Демонстрація роботи алгоритмів для знаходження частих елементів."""
    import random
    import time
    import sys
    from collections import Counter
    
    # Параметри експерименту
    n = 100000  # кількість елементів
    distinct = 1000  # кількість різних елементів
    zipf_param = 1.5  # параметр розподілу Zipf (більше значення = більша нерівномірність)
    k = 50  # кількість лічильників для алгоритмів
    
    # Генеруємо дані з розподілом Zipf (багато повторень найпопулярніших елементів)
    print(f"Генерація {n} елементів з розподілом Zipf (показник {zipf_param})...")
    
    # Визначення ваг для розподілу Zipf
    weights = [1.0 / (i ** zipf_param) for i in range(1, distinct + 1)]
    weight_sum = sum(weights)
    normalized_weights = [w / weight_sum for w in weights]
    
    # Генеруємо дані
    data = random.choices(range(distinct), weights=normalized_weights, k=n)
    
    # Точний підрахунок (Counter)
    start_time = time.time()
    counter = Counter(data)
    exact_time = time.time() - start_time
    
    # Алгоритм Misra-Gries
    mg = MisraGries(k)
    
    start_time = time.time()
    mg.process_batch(data)
    mg_time = time.time() - start_time
    
    # Алгоритм Space-Saving
    ss = SpaceSaving(k)
    
    start_time = time.time()
    for item in data:
        ss.add(item)
    ss_time = time.time() - start_time
    
    # Порівняння точності для топ-10 елементів
    print("\nПорівняння точності для топ-10 найчастіших елементів:")
    print("-" * 75)
    print("Елемент | Точно    | Misra-Gries | Відхилення | Space-Saving | Відхилення (похибка)")
    print("-" * 75)
    
    # Знаходимо топ-10 найчастіших елементів
    top_items = [item for item, _ in counter.most_common(10)]
    
    mg_error_sum = 0
    ss_error_sum = 0
    
    for item in top_items:
        exact_count = counter[item]
        mg_count = mg.counters.get(item, 0)
        ss_count, ss_error = ss.get_count(item)
        
        mg_error = mg_count - exact_count
        mg_error_percent = mg_error * 100 / exact_count if exact_count else 0
        mg_error_sum += abs(mg_error)
        
        ss_real_error = ss_count - exact_count
        ss_error_percent = ss_real_error * 100 / exact_count if exact_count else 0
        ss_error_sum += abs(ss_real_error)
        
        print(f"{item:7d} | {exact_count:8d} | {mg_count:11d} | {mg_error:+d} ({mg_error_percent:+.2f}%) | {ss_count:12d} | {ss_real_error:+d} ({ss_error_percent:+.2f}%) [{ss_error}]")
    
    mg_avg_error = mg_error_sum / len(top_items)
    ss_avg_error = ss_error_sum / len(top_items)
    
    print(f"\nСереднє абсолютне відхилення для топ-10:")
    print(f"Misra-Gries: {mg_avg_error:.2f}")
    print(f"Space-Saving: {ss_avg_error:.2f}")
    
    # Порівняння швидкості та пам'яті
    exact_memory = sys.getsizeof(counter)
    mg_memory = sys.getsizeof(mg.counters)
    ss_memory = sys.getsizeof(ss.counters) + sys.getsizeof(ss.errors)
    
    print("\nПорівняння швидкості та пам'яті:")
    print(f"Точний підрахунок: час={exact_time:.4f}с, пам'ять={exact_memory/1024:.2f}КБ")
    print(f"Misra-Gries: час={mg_time:.4f}с, пам'ять={mg_memory/1024:.2f}КБ (економія: {(1 - mg_memory/exact_memory) * 100:.2f}%)")
    print(f"Space-Saving: час={ss_time:.4f}с, пам'ять={ss_memory/1024:.2f}КБ (економія: {(1 - ss_memory/exact_memory) * 100:.2f}%)")
    
    # Статистика алгоритмів
    print("\nСтатистика Misra-Gries:")
    for key, value in mg.stats().items():
        print(f"{key}: {value}")
    
    print("\nСтатистика Space-Saving:")
    for key, value in ss.stats().items():
        print(f"{key}: {value}")
    
    # Перевірка знаходження частих елементів
    print("\nПеревірка знаходження частих елементів (поріг 1%):")
    threshold = 0.01  # 1%
    
    # Підраховуємо реальні часті елементи
    real_frequent = {item: count for item, count in counter.items() 
                    if count > threshold * n}
    
    # Часті елементи за Misra-Gries
    mg_frequent = mg.get_frequent_items(threshold)
    
    # Часті елементи за Space-Saving
    ss_frequent = {item: count for item, count, _ in ss.get_top_k() 
                  if count > threshold * n}
    
    print(f"Реальна кількість частих елементів: {len(real_frequent)}")
    print(f"Знайдено Misra-Gries: {len(mg_frequent)}")
    print(f"Знайдено Space-Saving: {len(ss_frequent)}")
    
    # Обчислюємо точність і повноту
    mg_correct = sum(1 for item in mg_frequent if item in real_frequent)
    mg_precision = mg_correct / len(mg_frequent) if mg_frequent else 0
    mg_recall = mg_correct / len(real_frequent) if real_frequent else 0
    
    ss_correct = sum(1 for item in ss_frequent if item in real_frequent)
    ss_precision = ss_correct / len(ss_frequent) if ss_frequent else 0
    ss_recall = ss_correct / len(real_frequent) if real_frequent else 0
    
    print(f"\nТочність Misra-Gries: {mg_precision:.4f}")
    print(f"Повнота Misra-Gries: {mg_recall:.4f}")
    print(f"F1-міра Misra-Gries: {2 * mg_precision * mg_recall / (mg_precision + mg_recall) if mg_precision + mg_recall > 0 else 0:.4f}")
    
    print(f"\nТочність Space-Saving: {ss_precision:.4f}")
    print(f"Повнота Space-Saving: {ss_recall:.4f}")
    print(f"F1-міра Space-Saving: {2 * ss_precision * ss_recall / (ss_precision + ss_recall) if ss_precision + ss_recall > 0 else 0:.4f}")

if __name__ == "__main__":
    main()