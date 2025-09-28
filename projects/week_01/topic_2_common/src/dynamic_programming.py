"""
Модуль реалізації алгоритмів динамічного програмування
Author: Educational Tutorial
Python version: 3.8+
"""

from typing import List, Tuple, Dict, Optional
from functools import lru_cache
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass


@dataclass
class DPItem:
    """Представлення предмета для задачі про рюкзак 0/1"""
    name: str
    weight: int
    value: int


class DynamicProgramming:
    """Клас для демонстрації алгоритмів динамічного програмування"""
    
    @staticmethod
    def fibonacci_naive(n: int) -> int:
        """
        Наївна рекурсивна реалізація чисел Фібоначчі
        Часова складність: O(2^n) - жахливо неефективна!
        """
        if n <= 1:
            return n
        return DynamicProgramming.fibonacci_naive(n - 1) + DynamicProgramming.fibonacci_naive(n - 2)
    
    @staticmethod
    @lru_cache(maxsize=None)
    def fibonacci_memoized(n: int) -> int:
        """
        Фібоначчі з мемоізацією (Top-Down DP)
        Часова складність: O(n)
        """
        if n <= 1:
            return n
        return DynamicProgramming.fibonacci_memoized(n - 1) + DynamicProgramming.fibonacci_memoized(n - 2)
        
    
    @staticmethod
    def fibonacci_tabulation(n: int) -> int:
        """
        Фібоначчі з табуляцією (Bottom-Up DP)
        Часова складність: O(n), Просторова складність: O(1)
        """
        if n <= 1:
            return n
        
        dp = [0] * (n + 1)
        dp[1] = 1
        
        for i in range(2, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        
        return dp[n]

    @staticmethod
    def knapsack_01(items: List[DPItem], capacity: int) -> Tuple[int, List[DPItem]]:
        """
        Розв'язує задачу про рюкзак 0/1 за допомогою DP
        
        Args:
            items: Список предметів
            capacity: Місткість рюкзака
            
        Returns:
            Tuple[int, List[DPItem]]: Максимальна цінність та список обраних предметів
        """
        n = len(items)
        if n == 0 or capacity == 0:
            return 0, []
        
        # Створюємо 2D таблицю DP
        dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
        
        # Заповнюємо таблицю знизу вгору
        for i in range(1, n + 1):
            for w in range(1, capacity + 1):
                # Поточний предмет
                current_item = items[i - 1]
                
                if current_item.weight <= w:
                    # Можемо взяти предмет - обираємо максимум
                    dp[i][w] = max(
                        dp[i - 1][w],  # Не беремо
                        dp[i - 1][w - current_item.weight] + current_item.value  # Беремо
                    )
                else:
                    # Не можемо взяти - копіюємо попереднє значення
                    dp[i][w] = dp[i - 1][w]
        
        # Відновлюємо розв'язок (backtracking)
        selected_items = []
        w = capacity
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i - 1][w]:
                selected_items.append(items[i - 1])
                w -= items[i - 1].weight
        
        return dp[n][capacity], selected_items
    
    @staticmethod
    def coin_change_dp(coins: List[int], amount: int) -> Tuple[int, List[int]]:
        """
        Розв'язує задачу про розмін монет за допомогою DP
        
        Args:
            coins: Список номіналів монет
            amount: Сума для розміну
            
        Returns:
            Tuple[int, List[int]]: Мінімальна кількість монет та список монет
        """
        if amount == 0:
            return 0, []
        
        # dp[i] = мінімальна кількість монет для набору суми i
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        parent = [-1] * (amount + 1)  # Для відновлення розв'язку
        
        for i in range(1, amount + 1):
            for coin in coins:
                if coin <= i and dp[i - coin] + 1 < dp[i]:
                    dp[i] = dp[i - coin] + 1
                    parent[i] = coin
        
        if dp[amount] == float('inf'):
            return -1, []  # Неможливо набрати суму
        
        # Відновлюємо розв'язок
        result_coins = []
        current = amount
        while current > 0:
            coin = parent[current]
            result_coins.append(coin)
            current -= coin
        
        return dp[amount], result_coins
    
    @staticmethod
    def longest_common_subsequence(str1: str, str2: str) -> Tuple[int, str]:
        """
        Знаходить найдовшу спільну підпослідовність двох рядків
        
        Args:
            str1, str2: Вхідні рядки
            
        Returns:
            Tuple[int, str]: Довжина LCS та сама LCS
        """
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Заповнюємо таблицю DP
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        # Відновлюємо LCS
        lcs = ""
        i, j = m, n
        while i > 0 and j > 0:
            if str1[i - 1] == str2[j - 1]:
                lcs = str1[i - 1] + lcs
                i -= 1
                j -= 1
            elif dp[i - 1][j] > dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
        
        return dp[m][n], lcs
    
    @staticmethod
    def edit_distance(str1: str, str2: str) -> Tuple[int, List[str]]:
        """
        Обчислює редакційну відстань (Левенштейна) між двома рядками
        
        Args:
            str1, str2: Вхідні рядки
            
        Returns:
            Tuple[int, List[str]]: Мінімальна кількість операцій та список операцій
        """
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Ініціалізація базових випадків
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Заповнюємо таблицю
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # Видалення
                        dp[i][j - 1],      # Вставка
                        dp[i - 1][j - 1]   # Заміна
                    )
        
        # Відновлюємо операції
        operations = []
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0 and str1[i - 1] == str2[j - 1]:
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                operations.append(f"Замінити '{str1[i - 1]}' на '{str2[j - 1]}'")
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                operations.append(f"Видалити '{str1[i - 1]}'")
                i -= 1
            else:
                operations.append(f"Вставити '{str2[j - 1]}'")
                j -= 1
        
        return dp[m][n], operations[::-1]


class DPVisualizer:
    """Клас для візуалізації алгоритмів динамічного програмування"""
    
    @staticmethod
    def visualize_fibonacci_complexity(max_n=20):
        """Порівнює складність різних підходів до обчислення Фібоначчі"""



        """Порівнює складність різних підходів до Фібоначчі"""
        import timeit
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Тестуємо для різних значень n
        test_values = list(range(1, max_n + 1))
        naive_times = []
        memo_times = []
        tab_times = []
        
        for n in test_values:
            # Наївна рекурсія (обмеження на час виконання)
            if n <= 15:
                time_naive = timeit.timeit(lambda: DynamicProgramming.fibonacci_naive(n), number=1)
                naive_times.append(time_naive)
            else:
                naive_times.append(None)

            # Мемоізація
            DynamicProgramming.fibonacci_memoized.cache_clear()
            time_memo = timeit.timeit(lambda: DynamicProgramming.fibonacci_memoized(n), number=100) / 100
            memo_times.append(time_memo)

            # Табуляція
            time_tab = timeit.timeit(lambda: DynamicProgramming.fibonacci_tabulation(n), number=100) / 100
            tab_times.append(time_tab)

        # Графік часу виконання
        valid_naive = [(i, t) for i, t in enumerate(naive_times) if t is not None]
        if valid_naive:
            naive_x, naive_y = zip(*valid_naive)
            ax1.plot([test_values[i] for i in naive_x], naive_y, 'r-o', label='Наївний O(2^n)')
        
        ax1.plot(test_values, memo_times, 'g-s', label='Мемоізація O(n)')
        ax1.plot(test_values, tab_times, 'b-^', label='Табуляція O(n)')
        ax1.set_xlabel('n')
        ax1.set_ylabel('Час виконання (сек)')
        ax1.set_title('Порівняння часу виконання')
        ax1.legend()
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Графік результатів
        results = [DynamicProgramming.fibonacci_tabulation(n) for n in test_values[:15]]
        ax2.plot(test_values[:15], results, 'purple', marker='o', linewidth=2)
        ax2.set_xlabel('n')
        ax2.set_ylabel('F(n)')
        ax2.set_title('Числа Фібоначчі')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def visualize_knapsack_table(items: List[DPItem], capacity: int):
        """Візуалізує заповнення таблиці для задачі про рюкзак 0/1"""
        n = len(items)
        dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
        
        # Заповнюємо таблицю (повторюємо логіку з основного алгоритму)
        for i in range(1, n + 1):
            for w in range(1, capacity + 1):
                current_item = items[i - 1]
                if current_item.weight <= w:
                    dp[i][w] = max(
                        dp[i - 1][w],
                        dp[i - 1][w - current_item.weight] + current_item.value
                    )
                else:
                    dp[i][w] = dp[i - 1][w]
        
        # Візуалізація
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Створюємо heatmap
        sns.heatmap(dp, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                   xticklabels=list(range(capacity + 1)),
                   yticklabels=['∅'] + [f"{item.name}\n(w:{item.weight}, v:{item.value})" 
                                       for item in items])
        
        ax.set_title('Таблиця DP для задачі про рюкзак 0/1')
        ax.set_xlabel('Місткість рюкзака')
        ax.set_ylabel('Предмети')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def visualize_coin_change(coins: List[int], amount: int):
        """Візуалізує розв'язок задачі про розмін монет"""
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        parent = [-1] * (amount + 1)
        
        for i in range(1, amount + 1):
            for coin in coins:
                if coin <= i and dp[i - coin] + 1 < dp[i]:
                    dp[i] = dp[i - coin] + 1
                    parent[i] = coin
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Лівий графік: DP таблиця
        amounts = list(range(amount + 1))
        min_coins = [dp[i] if dp[i] != float('inf') else 0 for i in amounts]
        
        bars = ax1.bar(amounts, min_coins, color='lightblue')
        ax1.set_xlabel('Сума')
        ax1.set_ylabel('Мінімальна кількість монет')
        ax1.set_title('DP розв\'язок задачі про розмін монет')
        ax1.grid(True, alpha=0.3)
        
        # Правий графік: розв'язок для конкретної суми
        if dp[amount] != float('inf'):
            num_coins, result_coins = DynamicProgramming.coin_change_dp(coins, amount)
            coin_counts = {}
            for coin in result_coins:
                coin_counts[coin] = coin_counts.get(coin, 0) + 1
            
            coin_types = list(coin_counts.keys())
            counts = list(coin_counts.values())
            
            ax2.bar(coin_types, counts, color='lightgreen')
            ax2.set_xlabel('Номінал монети')
            ax2.set_ylabel('Кількість')
            ax2.set_title(f'Розв\'язок для суми {amount}\nВсього монет: {num_coins}')
            
            # Додаємо підписи на стовпчики
            for i, count in enumerate(counts):
                ax2.text(coin_types[i], count + 0.05, str(count), 
                        ha='center', va='bottom', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'Неможливо\nнабрати суму', 
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=16, fontweight='bold', color='red')
            ax2.set_title(f'Розв\'язок для суми {amount}')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def visualize_lcs_table(str1: str, str2: str):
        """Візуалізує таблицю LCS"""
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Заповнюємо таблицю
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        # Візуалізація
        fig, ax = plt.subplots(figsize=(max(8, n), max(6, m)))
        
        sns.heatmap(dp, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['∅'] + list(str2),
                   yticklabels=['∅'] + list(str1))
        
        # Виділяємо шлях LCS
        length, lcs = DynamicProgramming.longest_common_subsequence(str1, str2)
        
        ax.set_title(f'Таблиця LCS\nРядки: "{str1}" та "{str2}"\nLCS: "{lcs}" (довжина: {length})')
        ax.set_xlabel('Символи другого рядка')
        ax.set_ylabel('Символи першого рядка')
        
        plt.tight_layout()
        plt.show()


class PerformanceComparator:
    """Клас для порівняння ефективності жадібних алгоритмів та ДП"""

    @staticmethod
    def compare_knapsack_approaches(greedy_items=None, dp_items=None, capacity=None):
        """Порівнює жадібний підхід та ДП для задачі про рюкзак"""
        from greedy_algorithms import GreedyAlgorithms, Item
        from dynamic_programming import DynamicProgramming, DPItem  # Припущено

        # Реалістичні предмети за замовчуванням
        greedy_items = greedy_items or [
            Item("Ноутбук", 2.0, 300),
            Item("Аптечка", 1.0, 90),
            Item("Пляшка води", 1.0, 30),
            Item("Книга", 1.5, 50),
            Item("Павербанк", 0.5, 80),
            Item("Спальний мішок", 3.5, 110),
        ]

        dp_items = dp_items or [
            DPItem(item.name, item.weight, item.value) for item in greedy_items
        ]

        capacity = capacity if capacity is not None else 5.0

        # Жадібний підхід (дробовий)
        greedy_value, greedy_selected = GreedyAlgorithms.fractional_knapsack(greedy_items, capacity)

        # ДП підхід (0/1)
        dp_value, dp_selected = DynamicProgramming.knapsack_01(dp_items, capacity)

        print("=== ПОРІВНЯННЯ ПІДХОДІВ ДО РЮКЗАКА ===")
        print(f"Місткість рюкзака: {capacity}")
        print("\nПредмети:")
        for item in greedy_items:
            print(f"  {item.name}: вага={item.weight}, цінність={item.value}, щільність={item.value_per_weight:.2f}")

        print(f"\n1. ЖАДІБНИЙ ПІДХІД (дробовий рюкзак):")
        print(f"   Максимальна цінність: {greedy_value:.2f}")
        print("   Обрані предмети:")
        for item, fraction in greedy_selected:
            print(f"     {item.name}: {fraction:.1%} ({item.value * fraction:.1f} цінності)")

        print(f"\n2. ДИНАМІЧНЕ ПРОГРАМУВАННЯ (0/1 рюкзак):")
        print(f"   Максимальна цінність: {dp_value}")
        print("   Обрані предмети:")
        for item in dp_selected:
            print(f"     {item.name}: 100% ({item.value} цінності)")

        return greedy_value, dp_value

    @staticmethod
    def compare_coin_change(coins1=None, amount1=None, coins2=None, amount2=None):
        """Порівнює жадібний підхід та ДП для розміну монет"""
        from .greedy_algorithms import GreedyAlgorithms
        from .dynamic_programming import DynamicProgramming  # Припущено

        # Канонічна система
        coins1 = coins1 or [25, 10, 5, 1]
        amount1 = amount1 if amount1 is not None else 30

        # Неканонічна система (жадібний працює неефективно)
        coins2 = coins2 or [4, 3, 1]
        amount2 = amount2 if amount2 is not None else 6

        greedy_coins1, greedy_count1 = GreedyAlgorithms.greedy_coin_change(coins1, amount1)
        dp_count1, dp_coins1 = DynamicProgramming.coin_change_dp(coins1, amount1)

        greedy_coins2, greedy_count2 = GreedyAlgorithms.greedy_coin_change(coins2, amount2)
        dp_count2, dp_coins2 = DynamicProgramming.coin_change_dp(coins2, amount2)

        print("=== ПОРІВНЯННЯ РОЗМІНУ МОНЕТ ===")

        print(f"\n1. КАНОНІЧНА СИСТЕМА: монети {coins1}, сума {amount1}")
        print(f"   Жадібний: {greedy_count1} монет {greedy_coins1}")
        print(f"   ДП:       {dp_count1} монет {dp_coins1}")
        print(f"   Результат: {'Однаковий!' if greedy_count1 == dp_count1 else 'Відрізняється!'}")

        print(f"\n2. НЕКАНОНІЧНА СИСТЕМА: монети {coins2}, сума {amount2}")
        print(f"   Жадібний: {greedy_count2} монет {greedy_coins2}")
        print(f"   ДП:       {dp_count2} монет {dp_coins2}")
        print(f"   Результат: {'ДП кращий!' if dp_count2 < greedy_count2 else 'Однаковий!'}")

        return (greedy_count1, dp_count1), (greedy_count2, dp_count2)
