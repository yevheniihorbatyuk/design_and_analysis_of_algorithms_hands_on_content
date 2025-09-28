"""
Модуль тестування алгоритмів
Author: Educational Tutorial
Python version: 3.8+

Цей модуль містить юніт-тести для перевірки коректності реалізованих алгоритмів.
"""

import unittest
from typing import List
import sys
import os

# Додаємо поточну директорію в sys.path для імпорту модулів
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from greedy_algorithms import (
    GreedyAlgorithms, HuffmanCoding, Activity, Item
)
from dynamic_programming import (
    DynamicProgramming, DPItem
)
from graph_algorithms import (
    GraphAlgorithms, TSPSolver, Edge
)


class TestGreedyAlgorithms(unittest.TestCase):
    """Тести для жадібних алгоритмів"""
    
    def test_activity_selection(self):
        """Тест задачі про вибір заявок"""
        activities = [
            Activity("A", 1, 4),
            Activity("B", 3, 5),
            Activity("C", 0, 6),
            Activity("D", 5, 7),
            Activity("E", 8, 9),
            Activity("F", 5, 9)
        ]
        
        selected, indices = GreedyAlgorithms.activity_selection(activities)
        
        # Перевіряємо, що результат коректний
        self.assertGreater(len(selected), 0)
        
        # Перевіряємо, що заходи не перетинаються
        for i in range(len(selected) - 1):
            self.assertLessEqual(selected[i].end_time, selected[i + 1].start_time)
    
    def test_fractional_knapsack(self):
        """Тест дробового рюкзака"""
        items = [
            Item("Gold", 10, 60),    # density: 6.0
            Item("Silver", 20, 100), # density: 5.0
            Item("Bronze", 30, 120)  # density: 4.0
        ]
        capacity = 50.0
        
        max_value, selected = GreedyAlgorithms.fractional_knapsack(items, capacity)
        
        # Перевіряємо основні властивості
        self.assertGreater(max_value, 0)
        self.assertGreater(len(selected), 0)
        
        # Перевіряємо, що не перевищуємо місткість
        total_weight = sum(item.weight * fraction for item, fraction in selected)
        self.assertLessEqual(total_weight, capacity + 1e-9)  # Допуск на обчислення з плаваючою комою
    
    def test_greedy_coin_change(self):
        """Тест жадібного розміну монет"""
        # Канонічна система
        coins = [25, 10, 5, 1]
        amount = 30
        
        result_coins, count = GreedyAlgorithms.greedy_coin_change(coins, amount)
        
        self.assertEqual(sum(result_coins), amount)
        self.assertGreater(count, 0)
    
    def test_huffman_coding(self):
        """Тест алгоритму Гаффмана"""
        text = "ABRACADABRA"
        huffman = HuffmanCoding(text)
        
        # Перевіряємо, що всі символи мають коди
        for char in set(text):
            self.assertIn(char, huffman.codes)
        
        # Перевіряємо кодування та декодування
        encoded = huffman.encode()
        self.assertIsInstance(encoded, str)
        self.assertTrue(all(c in '01' for c in encoded))
        
        # Перевіряємо статистику
        stats = huffman.get_compression_stats()
        self.assertGreater(stats['original_bits'], 0)
        self.assertGreater(stats['encoded_bits'], 0)


class TestDynamicProgramming(unittest.TestCase):
    """Тести для динамічного програмування"""
    
    def test_fibonacci_consistency(self):
        """Тест послідовності різних реалізацій Фібоначчі"""
        n = 10
        
        # Порівнюємо результати різних методів (для малих n)
        if n <= 15:
            naive_result = DynamicProgramming.fibonacci_naive(n)
        else:
            naive_result = None
        
        DynamicProgramming.fibonacci_memoized.cache_clear()
        memo_result = DynamicProgramming.fibonacci_memoized(n)
        tab_result = DynamicProgramming.fibonacci_tabulation(n)
        
        # Всі методи повинні давати однаковий результат
        if naive_result is not None:
            self.assertEqual(naive_result, memo_result)
            self.assertEqual(naive_result, tab_result)
        self.assertEqual(memo_result, tab_result)
        
        # Перевіряємо відомі значення
        self.assertEqual(DynamicProgramming.fibonacci_tabulation(5), 5)
        self.assertEqual(DynamicProgramming.fibonacci_tabulation(7), 13)
    
    def test_knapsack_01(self):
        """Тест задачі про рюкзак 0/1"""
        items = [
            DPItem("Item1", 1, 1000),
            DPItem("Item2", 4, 3000),
            DPItem("Item3", 2, 2000)
        ]
        capacity = 5
        
        max_value, selected = DynamicProgramming.knapsack_01(items, capacity)
        
        # Перевіряємо базові властивості
        self.assertGreaterEqual(max_value, 0)
        
        # Перевіряємо обмеження ваги
        total_weight = sum(item.weight for item in selected)
        self.assertLessEqual(total_weight, capacity)
        
        # Перевіряємо цінність
        total_value = sum(item.value for item in selected)
        self.assertEqual(total_value, max_value)
    
    def test_coin_change_dp(self):
        """Тест ДП розміну монет"""
        coins = [1, 3, 4]
        amount = 6
        
        min_coins, coin_list = DynamicProgramming.coin_change_dp(coins, amount)
        
        # Перевірямо коректність розв'язку
        self.assertGreater(min_coins, 0)
        self.assertEqual(sum(coin_list), amount)
        self.assertEqual(len(coin_list), min_coins)
        
        # Перевіряємо, що ДП дає кращий результат ніж жадібний для цього випадку
        from greedy_algorithms import GreedyAlgorithms
        greedy_coins, greedy_count = GreedyAlgorithms.greedy_coin_change([4, 3, 1], amount)
        self.assertLessEqual(min_coins, greedy_count)
    
    def test_lcs(self):
        """Тест найдовшої спільної підпослідовності"""
        str1 = "AGGTAB"
        str2 = "GXTXAYB"
        
        length, lcs = DynamicProgramming.longest_common_subsequence(str1, str2)
        
        # Перевіряємо базові властивості
        self.assertGreaterEqual(length, 0)
        self.assertEqual(len(lcs), length)
        
        # Перевіряємо, що LCS дійсно є підпослідовністю обох рядків
        if lcs:
            i, j = 0, 0
            for char in lcs:
                # Шукаємо символ у першому рядку
                while i < len(str1) and str1[i] != char:
                    i += 1
                self.assertLess(i, len(str1), f"Символ {char} не знайдено в {str1}")
                i += 1
                
                # Шукаємо символ у другому рядку
                while j < len(str2) and str2[j] != char:
                    j += 1
                self.assertLess(j, len(str2), f"Символ {char} не знайдено в {str2}")
                j += 1
    
    def test_edit_distance(self):
        """Тест редакційної відстані"""
        str1 = "kitten"
        str2 = "sitting"
        
        distance, operations = DynamicProgramming.edit_distance(str1, str2)
        
        # Перевіряємо базові властивості
        self.assertGreaterEqual(distance, 0)
        self.assertIsInstance(operations, list)
        
        # Відома відстань для цього прикладу
        self.assertEqual(distance, 3)


class TestGraphAlgorithms(unittest.TestCase):
    """Тести для графових алгоритмів"""
    
    def setUp(self):
        """Підготовка тестових даних"""
        self.test_graph = {
            'A': [('B', 4), ('C', 2)],
            'B': [('A', 4), ('C', 1), ('D', 5)],
            'C': [('A', 2), ('B', 1), ('D', 8), ('E', 10)],
            'D': [('B', 5), ('C', 8), ('E', 2)],
            'E': [('C', 10), ('D', 2)]
        }
    
    def test_dijkstra(self):
        """Тест алгоритму Дейкстри"""
        distances, previous = GraphAlgorithms.dijkstra(self.test_graph, 'A')
        
        # Перевіряємо, що відстань до початкової вершини = 0
        self.assertEqual(distances['A'], 0)
        
        # Перевіряємо, що всі відстані не від'ємні
        for dist in distances.values():
            self.assertGreaterEqual(dist, 0)
        
        # Перевіряємо відомі найкоротші шляхи
        self.assertEqual(distances['C'], 2)  # A -> C напряму
        self.assertEqual(distances['B'], 3)  # A -> C -> B
    
    def test_prim_mst(self):
        """Тест алгоритму Пріма"""
        mst_edges, total_weight = GraphAlgorithms.prim_mst(self.test_graph)
        
        # MST для n вершин має n-1 ребер
        self.assertEqual(len(mst_edges), len(self.test_graph) - 1)
        
        # Перевіряємо, що загальна вага > 0
        self.assertGreater(total_weight, 0)
        
        # Перевіряємо, що сума ваг ребер дорівнює загальній вазі
        calculated_weight = sum(edge.weight for edge in mst_edges)
        self.assertAlmostEqual(total_weight, calculated_weight, places=2)
    
    def test_kruskal_mst(self):
        """Тест алгоритму Крускала"""
        vertices = list(self.test_graph.keys())
        edges = []
        
        # Створюємо список ребер з графа
        for vertex, neighbors in self.test_graph.items():
            for neighbor, weight in neighbors:
                if vertex < neighbor:  # Уникаємо дублювання
                    edges.append(Edge(vertex, neighbor, weight))
        
        mst_edges, total_weight = GraphAlgorithms.kruskal_mst(vertices, edges)
        
        # MST для n вершин має n-1 ребер
        self.assertEqual(len(mst_edges), len(vertices) - 1)
        
        # Перевіряємо, що загальна вага > 0
        self.assertGreater(total_weight, 0)
    
    def test_mst_algorithms_consistency(self):
        """Тест на узгодженість результатів Пріма та Крускала"""
        vertices = list(self.test_graph.keys())
        edges = []
        
        for vertex, neighbors in self.test_graph.items():
            for neighbor, weight in neighbors:
                if vertex < neighbor:
                    edges.append(Edge(vertex, neighbor, weight))
        
        prim_edges, prim_weight = GraphAlgorithms.prim_mst(self.test_graph)
        kruskal_edges, kruskal_weight = GraphAlgorithms.kruskal_mst(vertices, edges)
        
        # Обидва алгоритми повинні давати MST з однаковою вагою
        self.assertAlmostEqual(prim_weight, kruskal_weight, places=2)
    
    def test_tsp_nearest_neighbor(self):
        """Тест жадібної евристики для TSP"""
        distances = {
            ('A', 'B'): 10,
            ('A', 'C'): 15,
            ('A', 'D'): 20,
            ('B', 'C'): 35,
            ('B', 'D'): 25,
            ('C', 'D'): 30
        }
        
        route, total_distance = TSPSolver.nearest_neighbor_tsp(distances, 'A')
        
        # Перевіряємо базові властивості
        self.assertGreater(len(route), 1)
        self.assertGreater(total_distance, 0)
        
        # Перевіряємо, що маршрут починається і закінчується в стартовому місті
        self.assertEqual(route[0], 'A')
        self.assertEqual(route[-1], 'A')
        
        # Перевіряємо, що відвідуємо всі міста
        unique_cities = set(route[:-1])  # Виключаємо останнє (повторення початкового)
        expected_cities = set()
        for city1, city2 in distances.keys():
            expected_cities.add(city1)
            expected_cities.add(city2)
        self.assertEqual(unique_cities, expected_cities)


class TestPerformanceAndIntegration(unittest.TestCase):
    """Інтеграційні тести та тести продуктивності"""
    
    def test_knapsack_comparison(self):
        """Порівняння жадібного та ДП підходів до рюкзака"""
        from greedy_algorithms import Item
        
        # Створюємо однакові предмети для обох підходів
        greedy_items = [
            Item("A", 1, 4),
            Item("B", 2, 3),
            Item("C", 3, 2),
            Item("D", 4, 1)
        ]
        
        dp_items = [
            DPItem("A", 1, 4),
            DPItem("B", 2, 3),
            DPItem("C", 3, 2),
            DPItem("D", 4, 1)
        ]
        
        capacity = 5
        
        # Жадібний підхід (дробовий)
        greedy_value, _ = GreedyAlgorithms.fractional_knapsack(greedy_items, capacity)
        
        # ДП підхід (0/1)
        dp_value, _ = DynamicProgramming.knapsack_01(dp_items, capacity)
        
        # Дробовий рюкзак повинен давати не гірший результат ніж 0/1
        self.assertGreaterEqual(greedy_value, dp_value)
    
    def test_coin_change_comparison(self):
        """Порівняння жадібного та ДП підходів до розміну монет"""
        # Канонічна система - результати повинні співпадати
        canonical_coins = [25, 10, 5, 1]
        amount = 30
        
        greedy_coins, greedy_count = GreedyAlgorithms.greedy_coin_change(canonical_coins, amount)
        dp_count, dp_coins = DynamicProgramming.coin_change_dp(canonical_coins, amount)
        
        self.assertEqual(greedy_count, dp_count)
        self.assertEqual(sum(greedy_coins), amount)
        self.assertEqual(sum(dp_coins), amount)
        
        # Неканонічна система - ДП повинен бути не гірший
        non_canonical_coins = [4, 3, 1]
        amount = 6
        
        greedy_coins2, greedy_count2 = GreedyAlgorithms.greedy_coin_change(non_canonical_coins, amount)
        dp_count2, dp_coins2 = DynamicProgramming.coin_change_dp(non_canonical_coins, amount)
        
        self.assertLessEqual(dp_count2, greedy_count2)
    
    def test_fibonacci_performance_scaling(self):
        """Тест масштабованості різних реалізацій Фібоначчі"""
        import time
        
        # Тестуємо для різних розмірів
        test_sizes = [10, 20, 30]
        
        for n in test_sizes:
            # Мемоізація
            DynamicProgramming.fibonacci_memoized.cache_clear()
            start_time = time.time()
            memo_result = DynamicProgramming.fibonacci_memoized(n)
            memo_time = time.time() - start_time
            
            # Табуляція
            start_time = time.time()
            tab_result = DynamicProgramming.fibonacci_tabulation(n)
            tab_time = time.time() - start_time
            
            # Результати повинні співпадати
            self.assertEqual(memo_result, tab_result)
            
            # Для великих n табуляція повинна бути швидшою
            if n >= 20:
                self.assertLess(tab_time, memo_time * 2)  # Допуск на варіації


class TestEdgeCases(unittest.TestCase):
    """Тести граничних випадків"""
    
    def test_empty_inputs(self):
        """Тест поведінки з порожніми вхідними даними"""
        # Порожні списки заходів
        selected, indices = GreedyAlgorithms.activity_selection([])
        self.assertEqual(len(selected), 0)
        self.assertEqual(len(indices), 0)
        
        # Порожні списки предметів
        value, items = GreedyAlgorithms.fractional_knapsack([], 10)
        self.assertEqual(value, 0.0)
        self.assertEqual(len(items), 0)
        
        # Рюкзак з нульовою місткістю
        test_items = [DPItem("A", 1, 10)]
        max_value, selected = DynamicProgramming.knapsack_01(test_items, 0)
        self.assertEqual(max_value, 0)
        self.assertEqual(len(selected), 0)
    
    def test_single_elements(self):
        """Тест з одним елементом"""
        # Один захід
        single_activity = [Activity("Single", 1, 3)]
        selected, indices = GreedyAlgorithms.activity_selection(single_activity)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].name, "Single")
        
        # Один предмет в рюкзаку
        single_item = [DPItem("Single", 2, 10)]
        max_value, selected = DynamicProgramming.knapsack_01(single_item, 5)
        self.assertEqual(max_value, 10)
        self.assertEqual(len(selected), 1)
    
    def test_fibonacci_base_cases(self):
        """Тест базових випадків Фібоначчі"""
        self.assertEqual(DynamicProgramming.fibonacci_tabulation(0), 0)
        self.assertEqual(DynamicProgramming.fibonacci_tabulation(1), 1)
        
        DynamicProgramming.fibonacci_memoized.cache_clear()
        self.assertEqual(DynamicProgramming.fibonacci_memoized(0), 0)
        self.assertEqual(DynamicProgramming.fibonacci_memoized(1), 1)


def run_all_tests():
    """Запуск всіх тестів з детальним звітом"""
    print("🧪 ЗАПУСК ТЕСТІВ ДЛЯ АЛГОРИТМІВ")
    print("=" * 50)
    
    # Створюємо test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Додаємо всі тестові класи
    test_classes = [
        TestGreedyAlgorithms,
        TestDynamicProgramming, 
        TestGraphAlgorithms,
        TestPerformanceAndIntegration,
        TestEdgeCases
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Запускаємо тести з детальним виводом
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Виводимо підсумок
    print("\n" + "=" * 50)
    print("📊 ПІДСУМОК ТЕСТУВАННЯ")
    print("=" * 50)
    print(f"Всього тестів: {result.testsRun}")
    print(f"Успішних: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Невдач: {len(result.failures)}")
    print(f"Помилок: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ НЕВДАЧІ:")
        for test, traceback in result.failures:
            print(f"  • {test}: {traceback.split('AssertionError: ')[-1].strip()}")
    
    if result.errors:
        print("\n💥 ПОМИЛКИ:")
        for test, traceback in result.errors:
            print(f"  • {test}: {traceback.split('Exception: ')[-1].strip()}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n✅ Успішність: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    
    # Перевіряємо наявність необхідних модулів
    try:
        from greedy_algorithms import GreedyAlgorithms
        from dynamic_programming import DynamicProgramming
        from graph_algorithms import GraphAlgorithms
        print("✅ Всі модулі успішно імпортовано")
    except ImportError as e:
        print(f"❌ Помилка імпорту: {e}")
        print("Переконайтесь, що всі файли модулів знаходяться в тій же директорії")
        sys.exit(1)
    
    # Запускаємо тести
    success = run_all_tests()
    
    if success:
        print("\n🎉 ВСІ ТЕСТИ ПРОЙШЛИ УСПІШНО!")
        print("Алгоритми реалізовано коректно та готові до демонстрації.")
    else:
        print("\n⚠️ ДЕЯКІ ТЕСТИ НЕ ПРОЙШЛИ!")
        print("Рекомендується перевірити реалізацію перед демонстрацією.")
        sys.exit(1)