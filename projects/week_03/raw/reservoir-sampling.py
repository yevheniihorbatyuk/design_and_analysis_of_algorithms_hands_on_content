import random
from typing import List, TypeVar, Iterator, Generic, Optional

T = TypeVar('T')

class ReservoirSampling(Generic[T]):
    """
    Реалізація алгоритму резервуарної вибірки (Reservoir Sampling).
    
    Алгоритм дозволяє отримати рівномірну випадкову вибірку фіксованого розміру
    з потоку даних, розмір якого заздалегідь невідомий.
    
    Time Complexity:
        - Обробка нового елемента: O(1) в середньому
        - Загальна складність: O(n) для n елементів
    Space Complexity:
        - O(k), де k - розмір резервуару
    """
    
    def __init__(self, reservoir_size: int):
        """
        Ініціалізує резервуарну вибірку.
        
        Args:
            reservoir_size: розмір резервуару (кількість елементів у вибірці)
        """
        if reservoir_size <= 0:
            raise ValueError("Розмір резервуару має бути більшим за нуль.")
        
        self.reservoir_size = reservoir_size
        self.reservoir: List[T] = []
        self.count = 0  # Кількість оброблених елементів
    
    def add(self, item: T) -> None:
        """
        Додає новий елемент до потоку і оновлює резервуар.
        
        Args:
            item: елемент для додавання
        """
        self.count += 1
        
        # Якщо резервуар не повний, просто додаємо елемент
        if len(self.reservoir) < self.reservoir_size:
            self.reservoir.append(item)
        else:
            # Вирішуємо, чи замінити елемент у резервуарі
            # Ймовірність заміни: k/n, де k=reservoir_size, n=count
            j = random.randint(0, self.count - 1)
            if j < self.reservoir_size:
                self.reservoir[j] = item
    
    def process_stream(self, stream: Iterator[T], max_items: Optional[int] = None) -> None:
        """
        Обробляє потік даних і заповнює резервуар.
        
        Args:
            stream: ітератор з даними
            max_items: максимальна кількість елементів для обробки (None для необмеженої)
        """
        items_processed = 0
        for item in stream:
            self.add(item)
            items_processed += 1
            if max_items is not None and items_processed >= max_items:
                break
    
    def get_sample(self) -> List[T]:
        """
        Повертає поточну вибірку.
        
        Returns:
            Список елементів у резервуарі.
        """
        return self.reservoir.copy()
    
    def reset(self) -> None:
        """Скидає стан резервуару."""
        self.reservoir = []
        self.count = 0

def main():
    """Демонстрація роботи алгоритму резервуарної вибірки."""
    # Генеруємо великий потік даних (для прикладу використовуємо числа)
    stream_size = 1_000_000
    sample_size = 10
    
    # Створюємо резервуарну вибірку і заповнюємо її з потоку
    reservoir = ReservoirSampling[int](sample_size)
    
    # Демонстрація на великому потоці
    print(f"Беремо вибірку {sample_size} елементів з потоку розміром {stream_size}:")
    reservoir.process_stream(range(stream_size))
    print(f"Резервуарна вибірка: {reservoir.get_sample()}")
    
    # Статистична перевірка (всі елементи мають бути приблизно рівноймовірними)
    buckets = 10
    frequencies = {i: 0 for i in range(buckets)}
    num_trials = 10000
    
    for _ in range(num_trials):
        reservoir.reset()
        stream = (random.randint(0, buckets-1) for _ in range(stream_size))
        reservoir.process_stream(stream)
        for item in reservoir.get_sample():
            frequencies[item] += 1
    
    print("\nСтатистична перевірка (частоти для 10 елементів):")
    for bucket, freq in frequencies.items():
        expected = num_trials * sample_size / buckets
        print(f"Елемент {bucket}: {freq} (очікувано ~{expected:.0f})")

if __name__ == "__main__":
    main()
