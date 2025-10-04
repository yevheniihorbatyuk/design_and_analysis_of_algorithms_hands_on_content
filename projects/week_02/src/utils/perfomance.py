# src/utils/performance.py

import pandas as pd

class PerformanceTester:
    """Клас для тестування та порівняння продуктивності алгоритмів."""
    @staticmethod
    def compare(graph, start_node, end_node, algorithms):
        """
        Порівнює кілька алгоритмів на одній задачі.
        
        Args:
            graph: Граф NetworkX.
            start_node: Початкова вершина.
            end_node: Кінцева вершина.
            algorithms (dict): Словник, де ключ - назва, а значення - кортеж 
                               (функція_алгоритму, dict_з_аргументами).
        
        Returns:
            list: Список словників з результатами для кожного алгоритму.
        """
        results = []
        for name, (func, kwargs) in algorithms.items():
            result = func(graph, start_node, end_node, **kwargs)
            results.append(result)
        return results