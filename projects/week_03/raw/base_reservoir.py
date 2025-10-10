import random
from typing import List, Iterator, TypeVar

T = TypeVar('T')

def reservoir_sampling(stream: Iterator[T], k: int) -> List[T]:
    """
    Реалізує алгоритм резервуарної вибірки.
    
    Args:
        stream: Ітератор елементів
        k: Розмір вибірки
    
    Returns:
        List[T]: Випадкова вибірка розміру k
    """
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir

if __name__ == "__main__":
    # Тестування
    stream = range(1, 101)  # Потік чисел від 1 до 100
    k = 10  # Розмір вибірки
    sample = reservoir_sampling(stream, k)
    print("Випадкова вибірка:", sample)