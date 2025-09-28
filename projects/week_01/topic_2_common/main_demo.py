"""
Основний демонстраційний скрипт для жадібних алгоритмів та динамічного програмування
Author: Educational Tutorial
Python version: 3.8+

Цей скрипт демонструє всі алгоритми з відео-туторіалу з інтерактивними прикладами та візуалізацією.
"""

import sys
from typing import List
import matplotlib.pyplot as plt

# Імпортуємо наші модулі
from greedy_algorithms import (
    GreedyAlgorithms, HuffmanCoding, GreedyVisualizer,
    Activity, Item
)
from dynamic_programming import (
    DynamicProgramming, DPVisualizer, PerformanceComparator,
    DPItem
)
from graph_algorithms import (
    GraphAlgorithms, GraphVisualizer, TSPSolver,
    Edge
)


def demo_activity_selection(activities=None):
    """Демонстрація задачі про вибір заявок"""
    print("=" * 60)
    print("ДЕМОНСТРАЦІЯ: Задача про вибір заявок (Activity Selection)")
    print("=" * 60)
    
    # Створюємо тестові дані
    activities = activities or [
        Activity("Лекція A", 1, 4),
        Activity("Лекція B", 3, 5),
        Activity("Лекція C", 0, 6),
        Activity("Лекція D", 5, 7),
        Activity("Лекція E", 8, 9),
        Activity("Лекція F", 5, 9)
    ]


    
    # Жадібний алгоритм
    selected, indices = GreedyAlgorithms.activity_selection(activities)

    print(f"\nВсього активностей: {len(activities)}")
    print("Повний розклад:")
    for i, activity in enumerate(activities):
        status = "✓ ОБРАНО" if i in indices else "✗ пропущено"
        print(f"  {i+1}. {activity.name:30} ({activity.start_time:>4.1f} – {activity.end_time:<4.1f})  {status}")

    print(f"\nОбрано {len(selected)} активностей для розкладу без конфліктів:")
    for activity in selected:
        print(f"  • {activity.name} ({activity.start_time:.1f}-{activity.end_time:.1f})")

    print(f"\nСтратегія: сортування за часом ЗАКІНЧЕННЯ (ранше закінчується – краще)")
    print("Часова складність: O(n log n)")
    
    # Візуалізація
    GreedyVisualizer.visualize_activity_selection(activities, selected)

    return selected



def demo_fractional_knapsack(items=None, capacity=None):
    """Демонстрація задачі про дробовий рюкзак (Fractional Knapsack)"""
    print("=" * 60)
    print("ДЕМОНСТРАЦІЯ: Задача про дробовий рюкзак (Fractional Knapsack)")
    print("=" * 60)

    # Якщо не передано – використовуємо стандартні тестові предмети
    items = items or [
        Item("Золото", 10, 60),      # щільність: 6.0
        Item("Срібло", 20, 100),     # щільність: 5.0  
        Item("Діаманти", 30, 120),   # щільність: 4.0
        Item("Бронза", 15, 45),      # щільність: 3.0
        Item("Залізо", 25, 50)       # щільність: 2.0
    ]

    capacity = capacity if capacity is not None else 50.0

    # Виконуємо жадібний алгоритм
    max_value, selected = GreedyAlgorithms.fractional_knapsack(items, capacity)

    print(f"Місткість рюкзака: {capacity}")
    print("\nПредмети (відсортовані за щільністю цінність/вага):")
    sorted_items = sorted(items, reverse=True)  # За спаданням щільності
    for item in sorted_items:
        print(f"  {item.name}: вага={item.weight}, цінність={item.value}, щільність={item.value_per_weight:.2f}")

    print(f"\nРезультат жадібного алгоритму:")
    print(f"  Максимальна цінність: {max_value:.2f}")
    print("  Обрані предмети:")

    total_weight = 0
    for item, fraction in selected:
        weight_taken = item.weight * fraction
        value_taken = item.value * fraction
        total_weight += weight_taken
        print(f"    {item.name}: {fraction:.1%} (вага: {weight_taken:.1f}, цінність: {value_taken:.1f})")

    print(f"  Загальна вага: {total_weight:.1f}/{capacity}")
    print(f"\nСтратегія: сортування за щільністю (цінність/вага)")
    print("Часова складність: O(n log n)")

    # Візуалізація
    GreedyVisualizer.visualize_knapsack(items, selected, capacity)

    return max_value, selected


def demo_huffman_coding(text=None):
    """Демонстрація алгоритму Гаффмана"""
    print("=" * 60)
    print("ДЕМОНСТРАЦІЯ: Алгоритм Гаффмана (Huffman Coding)")
    print("=" * 60)

    text = text or "ABRACADABRA"
    huffman = HuffmanCoding(text)

    print(f"Оригінальний текст: '{text}'")
    print(f"Довжина: {len(text)} символів")

    print(f"\nТаблиця частот:")
    for char, freq in sorted(huffman.freq_table.items()):
        print(f"  '{char}': {freq} разів")

    print(f"\nКоди Гаффмана:")
    for char, code in sorted(huffman.codes.items()):
        print(f"  '{char}': {code}")

    encoded = huffman.encode()
    print(f"\nЗакодований текст: {encoded}")
    print(f"Довжина закодованого: {len(encoded)} біт")

    stats = huffman.get_compression_stats()
    print(f"\nСтатистика стиснення:")
    print(f"  ASCII (8 біт/символ): {stats['original_bits']} біт")
    print(f"  Гаффман: {stats['encoded_bits']} біт")
    print(f"  Коефіцієнт стиснення: {stats['compression_ratio']:.3f}")
    print(f"  Економія місця: {stats['space_saved']:.1%}")

    print(f"\nСтратегія: жадібне об'єднання найрідших символів")
    print("Часова складність: O(n log n)")

    GreedyVisualizer.visualize_huffman_tree(huffman)
    return huffman



def demo_fibonacci_comparison(n=None, visualize=True):
    """Демонстрація порівняння підходів до Фібоначчі"""
    print("=" * 60)
    print("ДЕМОНСТРАЦІЯ: Числа Фібоначчі (Наївна рекурсія vs ДП)")
    print("=" * 60)

    n = n if n is not None else 10
    print(f"Обчислюємо F({n}) трьома способами:")

    if n <= 20:
        naive_result = DynamicProgramming.fibonacci_naive(n)
        print(f"  1. Наївна рекурсія: F({n}) = {naive_result}")
        print(f"     Часова складність: O(2^n) - ЖАХЛИВО!")

    DynamicProgramming.fibonacci_memoized.cache_clear()
    memo_result = DynamicProgramming.fibonacci_memoized(n)
    print(f"  2. Мемоізація (Top-Down): F({n}) = {memo_result}")
    print(f"     Часова складність: O(n)")

    tab_result = DynamicProgramming.fibonacci_tabulation(n)
    print(f"  3. Табуляція (Bottom-Up): F({n}) = {tab_result}")
    print(f"     Часова складність: O(n), Просторова: O(1)")

    print(f"\nПершні {n+1} чисел Фібоначчі:")
    fib_sequence = [DynamicProgramming.fibonacci_tabulation(i) for i in range(n+1)]
    print(f"  {fib_sequence}")

    if visualize:
        DPVisualizer.visualize_fibonacci_complexity(n)

    return fib_sequence

def demo_knapsack_01(items=None, capacity=None):
    """Демонстрація задачі про рюкзак 0/1"""
    print("=" * 60)
    print("ДЕМОНСТРАЦІЯ: Задача про рюкзак 0/1 (Dynamic Programming)")
    print("=" * 60)

    items = items or [
        DPItem("Телефон", 1, 1000),
        DPItem("Ноутбук", 4, 3000),
        DPItem("Камера", 2, 2000),
        DPItem("Книга", 1, 100),
        DPItem("Планшет", 3, 1500)
    ]
    capacity = capacity if capacity is not None else 5

    print(f"Місткість рюкзака: {capacity}")
    print("Доступні предмети:")
    for i, item in enumerate(items):
        print(f"  {i+1}. {item.name}: вага={item.weight}, цінність={item.value}")

    max_value, selected_items = DynamicProgramming.knapsack_01(items, capacity)

    print(f"\nРезультат динамічного програмування:")
    print(f"  Максимальна цінність: {max_value}")
    print("  Обрані предмети:")
    total_weight = 0
    for item in selected_items:
        total_weight += item.weight
        print(f"    • {item.name} (вага: {item.weight}, цінність: {item.value})")

    print(f"  Загальна вага: {total_weight}/{capacity}")
    print(f"\nЧому жадібний підхід тут НЕ працює:")
    print("  Жадібний би обрав за щільністю (цінність/вага)")
    items_by_density = sorted(items, key=lambda x: x.value/x.weight, reverse=True)
    print("  Порядок за щільністю:")
    for item in items_by_density:
        print(f"    {item.name}: щільність = {item.value/item.weight:.1f}")

    print("\nСтратегія ДП: розглядаємо ВСІ можливості через підзадачі")
    print("Часова складність: O(n × W), де W - місткість")

    DPVisualizer.visualize_knapsack_table(items, capacity)
    return max_value, selected_items



def demo_coin_change(coins=None, amount=None):
    """Демонстрація задачі про розмін монет"""
    print("=" * 60)
    print("ДЕМОНСТРАЦІЯ: Задача про розмін монет (Жадібний vs ДП)")
    print("=" * 60)

    PerformanceComparator.compare_coin_change()

    coins = coins or [4, 3, 1]
    amount = amount if amount is not None else 6

    print(f"\nДетальний аналіз для монет {coins} і суми {amount}:")

    min_coins, coin_list = DynamicProgramming.coin_change_dp(coins, amount)
    print(f"ДП знайшов оптимальний розв'язок: {min_coins} монет {coin_list}")

    DPVisualizer.visualize_coin_change(coins, amount)
    return min_coins, coin_list


def demo_graph_algorithms(graph=None, start_node=None):
    """Демонстрація графових алгоритмів"""
    print("=" * 60)
    print("ДЕМОНСТРАЦІЯ: Жадібні алгоритми на графах")
    print("=" * 60)

    graph = graph or {
        'A': [('B', 4), ('C', 2)],
        'B': [('A', 4), ('C', 1), ('D', 5)],
        'C': [('A', 2), ('B', 1), ('D', 8), ('E', 10)],
        'D': [('B', 5), ('C', 8), ('E', 2)],
        'E': [('C', 10), ('D', 2)]
    }
    start_node = start_node or 'A'

    print("Тестовий граф:")
    for vertex, edges in graph.items():
        neighbors = ", ".join([f"{neighbor}({weight})" for neighbor, weight in edges])
        print(f"  {vertex}: {neighbors}")

    print(f"\n1. АЛГОРИТМ ДЕЙКСТРИ (з вершини {start_node}):")
    distances, previous = GraphAlgorithms.dijkstra(graph, start_node)
    print("   Найкоротші відстані:")
    for vertex, dist in distances.items():
        print(f"     {start_node} → {vertex}: {dist if dist != float('inf') else '∞'}")

    GraphVisualizer.visualize_dijkstra(graph, start_node)

    print(f"\n2. МІНІМАЛЬНІ ОСТОВНІ ДЕРЕВА:")
    prim_edges, prim_weight = GraphAlgorithms.prim_mst(graph)
    print(f"   Алгоритм Пріма: вага MST = {prim_weight}")
    for edge in prim_edges:
        print(f"     {edge.start} - {edge.end}: {edge.weight}")

    GraphVisualizer.visualize_mst_comparison(graph)
    return distances, prim_edges


def demo_tsp(distances=None, start_city=None):
    """Демонстрація жадібної евристики для TSP"""
    print("=" * 60)
    print("ДЕМОНСТРАЦІЯ: Задача Комівояжера (TSP) - Жадібна евристика")
    print("=" * 60)

    distances = distances or {
        ('Київ', 'Львів'): 540,
        ('Київ', 'Одеса'): 475,
        ('Київ', 'Харків'): 480,
        ('Львів', 'Одеса'): 790,
        ('Львів', 'Харків'): 1050,
        ('Одеса', 'Харків'): 730
    }
    start_city = start_city or 'Київ'

    print("Матриця відстаней між містами:")
    cities = {city for pair in distances.keys() for city in pair}
    for city1 in sorted(cities):
        for city2 in sorted(cities):
            if city1 != city2:
                dist = distances.get((city1, city2)) or distances.get((city2, city1))
                print(f"  {city1} → {city2}: {dist} км")

    route, total_distance = TSPSolver.nearest_neighbor_tsp(distances, start_city)

    print(f"\nЖадібна евристика 'Найближчий сусід' (початок: {start_city}):")
    print(f"  Маршрут: {' → '.join(route)}")
    print(f"  Загальна відстань: {total_distance} км")

    print(f"\nУВАГА: Це евристика для NP-повної задачі!")
    print("  ✓ Швидко знаходить 'достатньо хороший' розв'язок")
    print("  ✗ Не гарантує оптимальність")

    GraphVisualizer.visualize_tsp_solution(distances, start_city)
    return route, total_distance



def demo_comprehensive_comparison():
    """Комплексне порівняння жадібних алгоритмів та ДП"""
    print("=" * 60)
    print("КОМПЛЕКСНЕ ПОРІВНЯННЯ: Жадібні vs Динамічне Програмування")
    print("=" * 60)
    
    # Порівняння підходів до рюкзака
    print("\n1. ПОРІВНЯННЯ ПІДХОДІВ ДО РЮКЗАКА:")
    greedy_value, dp_value = PerformanceComparator.compare_knapsack_approaches()
    
    # Порівняння розміну монет
    print("\n2. ПОРІВНЯННЯ РОЗМІНУ МОНЕТ:")
    canonical_result, non_canonical_result = PerformanceComparator.compare_coin_change()
    
    # Загальні висновки
    print("\n" + "="*60)
    print("ЗАГАЛЬНІ ВИСНОВКИ:")
    print("="*60)
    
    print("\n🏃 ЖАДІБНІ АЛГОРИТМИ:")
    print("  ✓ Швидкі та прості в реалізації")  
    print("  ✓ Ефективні за пам'яттю")
    print("  ✓ Підходять для задач з 'властивістю жадібного вибору'")
    print("  ✗ Не завжди дають оптимальний результат")
    print("  ✗ Важко довести коректність")
    
    print("\n🧠 ДИНАМІЧНЕ ПРОГРАМУВАННЯ:")
    print("  ✓ Гарантує оптимальний розв'язок (якщо застосовне)")
    print("  ✓ Потужний для задач з підзадачами, що перекриваються")
    print("  ✓ Добре вивчена теорія застосування")
    print("  ✗ Може потребувати багато пам'яті")
    print("  ✗ Складніший в реалізації та аналізі")
    
    print("\n🎯 КОЛИ ЩО ВИКОРИСТОВУВАТИ:")
    print("  • Жадібний → коли доведена властивість жадібного вибору")
    print("  • Жадібний → для швидких наближених розв'язків NP-повних задач")
    print("  • ДП → коли є оптимальна підструктура + підзадачі перекриваються")
    print("  • ДП → коли потрібен гарантовано оптимальний результат")
    
    return {
        'knapsack_comparison': (greedy_value, dp_value),
        'coin_comparison': (canonical_result, non_canonical_result)
    }


def main():
    """Головна функція для запуску всіх демонстрацій"""
    print("🎓 ЖАДІБНІ АЛГОРИТМИ ТА ДИНАМІЧНЕ ПРОГРАМУВАННЯ")
    print("📚 Навчальний туторіал з демонстраційними прикладами")
    print("="*80)
    
    # Меню вибору демонстрацій
    demos = {
        '1': ("Задача про вибір заявок (Activity Selection)", demo_activity_selection),
        '2': ("Дробовий рюкзак (Fractional Knapsack)", demo_fractional_knapsack), 
        '3': ("Алгоритм Гаффмана (Huffman Coding)", demo_huffman_coding),
        '4': ("Числа Фібоначчі (Fibonacci Comparison)", demo_fibonacci_comparison),
        '5': ("Рюкзак 0/1 (0/1 Knapsack DP)", demo_knapsack_01),
        '6': ("Розмін монет (Coin Change)", demo_coin_change),
        '7': ("Графові алгоритми (Dijkstra, MST)", demo_graph_algorithms),
        '8': ("Задача комівояжера (TSP)", demo_tsp),
        '9': ("Комплексне порівняння", demo_comprehensive_comparison),
        'a': ("ВСІ ДЕМОНСТРАЦІЇ ПІДРЯД", None)
    }
    
    print("\nОберіть демонстрацію:")
    for key, (title, _) in demos.items():
        print(f"  {key}. {title}")
    
    choice = input("\nВаш вибір (або Enter для всіх): ").lower().strip()
    
    if not choice:
        choice = 'a'
    
    print("\n")
    
    try:
        if choice == 'a':
            # Запускаємо всі демонстрації підряд
            results = {}
            for key in sorted(demos.keys()):
                if key != 'a' and demos[key][1] is not None:
                    print(f"\n{'='*20} ДЕМОНСТРАЦІЯ {key} {'='*20}")
                    results[key] = demos[key][1]()
                    input("\nНатисніть Enter для продовження...")
            
            print(f"\n🎉 Всі демонстрації завершено успішно!")
            
        elif choice in demos and demos[choice][1] is not None:
            # Запускаємо конкретну демонстрацію
            result = demos[choice][1]()
            print(f"\n✅ Демонстрацію '{demos[choice][0]}' завершено!")
            
        else:
            print("❌ Невірний вибір!")
            return
            
    except KeyboardInterrupt:
        print(f"\n\n⏹️ Демонстрацію перервано користувачем")
    except Exception as e:
        print(f"\n❌ Помилка під час демонстрації: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("🎓 Дякуємо за увагу до навчального матеріалу!")
    print("📖 Рекомендуємо поекспериментувати з кодом самостійно")
    print("="*80)


if __name__ == "__main__":
    # Налаштування для кращого відображення графіків
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    main()