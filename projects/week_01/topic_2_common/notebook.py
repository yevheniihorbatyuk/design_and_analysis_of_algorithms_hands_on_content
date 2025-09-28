# %% [markdown]
# ## Imports

# %%
from main_demo import (
    demo_activity_selection,
    demo_fractional_knapsack,
    demo_huffman_coding,
    demo_fibonacci_comparison,
    demo_knapsack_01,
    demo_coin_change,
    demo_graph_algorithms,
    demo_tsp,
    demo_comprehensive_comparison,
    main
    )
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

# %% [markdown]
# ## Задача про вибір заявок

# %%
demo_activity_selection()

# %%
# Реалістичний список подій з назвами та часом (у годинах)
activities = [
    Activity("Ранкова пробіжка", 6.0, 7.0),
    Activity("Сніданок з родиною", 7.0, 8.0),
    Activity("Робоча зустріч (Zoom)", 8.5, 9.5),
    Activity("Глибока робота над проектом", 9.0, 11.5),
    Activity("Прогулянка/кава-брейк", 11.5, 12.0),
    Activity("Онлайн-лекція", 12.0, 13.0),
    Activity("Обід", 13.0, 14.0),
    Activity("Зустріч з клієнтом", 13.5, 14.5),
    Activity("Читання / навчання", 14.5, 15.5),
    Activity("Спортзал", 16.0, 17.0),
    Activity("Покупки / справи", 17.0, 18.0),
    Activity("Вечеря", 18.5, 19.5),
    Activity("Перегляд фільму", 20.0, 22.0),
    Activity("Медитація / підготовка до сну", 22.0, 22.5)
]

demo_activity_selection(activities)

# %% [markdown]
# ## Задача 2: Задача про дробовий рюкзак (Fractional Knapsack)

# %%
demo_fractional_knapsack()

# %%
items = [
    Item("Ноутбук", 2.0, 300),         # дуже корисний, легкий
    Item("Пляшка води", 1.0, 30),      # життєво важлива
    Item("Книга", 1.5, 50),            # освітня цінність
    Item("Павербанк", 0.5, 80),        # багато користі при малій вазі
    Item("Куртка", 2.5, 60),           # потрібна при поганій погоді
    Item("Аптечка", 1.0, 90),          # критична для безпеки
    Item("Їжа (перекус)", 1.2, 40),    # енергетична підтримка
    Item("Навушники", 0.3, 40),        # невелика вага, помірна цінність
    Item("Планшет", 0.7, 150),         # альтернатива ноутбуку
    Item("Термобілизна", 1.3, 55),     # комфорт в умовах холоду
    Item("Записник + ручка", 0.4, 25), # низька вага, деяка користь
    Item("Ліхтарик", 0.6, 35),         # важливий у темряві
    Item("Зарядні кабелі", 0.2, 20),   # потрібні, але неважкі
    Item("Фотоапарат", 1.8, 120),      # висока цінність для блогера
    Item("Спальний мішок", 3.5, 110)   # важкий, але критично корисний у поході
]

capacity = 20

demo_fractional_knapsack(items, capacity)

# %% [markdown]
# ## "Обмеження Жадібних Алгоритмів".
# 
# - Приклад: "Задача про розмін монет"

# %%
PerformanceComparator.compare_coin_change()

# %%
cases = [
    # ✅ 1. Класичний випадок (канонічна система)
    {
        'coins1': [25, 10, 5, 1],
        'amount1': 30,
        'coins2': [10, 5, 1],
        'amount2': 28
    },
    # ❌ 2. Неканонічна система з "пасткою"
    {
        'coins1': [9, 6, 1],
        'amount1': 11,
        'coins2': [4, 3, 1],
        'amount2': 6
    },
    # ❌ 3. Випадок із неефективним greedy
    {
        'coins1': [7, 5, 1],
        'amount1': 18,
        'coins2': [9, 4, 1],
        'amount2': 15
    },
    # 🧠 4. Нестандартна валюта
    {
        'coins1': [1, 7, 10, 22],
        'amount1': 29,
        'coins2': [1, 3, 4],
        'amount2': 6
    },
    # ⚠️ 5. Ситуація, де greedy може бути правильним, але не завжди
    {
        'coins1': [1, 3, 4],
        'amount1': 5,
        'coins2': [1, 3, 4],
        'amount2': 6
    }
]

for i, param in enumerate(cases, 1):
    print(f"\n--- ПРИКЛАД {i} ---")
    PerformanceComparator.compare_coin_change(**param)


# %% [markdown]
# ## Алгоритм Гаффмана (Huffman Coding)

# %%
demo_huffman_coding()

# %%
text = "SHE SELLS SEA SHELLS BY THE SEA SHORE AND SHE STILL SELLS THEM SURELY"

demo_huffman_coding(text)

# %% [markdown]
# ## Числа Фібоначчі

# %%
demo_fibonacci_comparison()


# %%
n = 100

demo_fibonacci_comparison(n)


# %% [markdown]
# ## Задача про рюкзак (0/1 Knapsack Problem)

# %%

demo_knapsack_01()


# %%


items_01 = [
    DPItem("Ноутбук", 3, 2000),         # робота, висока цінність
    DPItem("Аптечка", 1, 1000),         # виживання
    DPItem("Термобілизна", 2, 700),     # тепло в польових умовах
    DPItem("Павербанк", 1, 800),        # живлення
    DPItem("Ліхтарик", 1, 500),         # орієнтація вночі
    DPItem("Фотоапарат", 2, 1200),      # для фіксації подій
    DPItem("Їжа", 3, 900),              # енергія
    DPItem("Книга", 1, 300),            # психологічна підтримка
]


capacity_01 = 8

demo_knapsack_01(items=items_01, capacity=capacity_01)


# %% [markdown]
# ## Алгоритм Дейкстри та Пріма/Крускала

# %%

demo_graph_algorithms()


# %%
graph = {
    'A': [('B', 2), ('C', 5), ('D', 1)],
    'B': [('A', 2), ('E', 3), ('F', 7)],
    'C': [('A', 5), ('F', 2), ('G', 6)],
    'D': [('A', 1), ('G', 1)],
    'E': [('B', 3), ('H', 4)],
    'F': [('B', 7), ('C', 2), ('H', 3)],
    'G': [('C', 6), ('D', 1), ('H', 2)],
    'H': [('E', 4), ('F', 3), ('G', 2)]
}


demo_graph_algorithms(graph=graph, start_node='A')

# %% [markdown]
# ## TSP

# %%

demo_tsp()


# %%
distances = {
    ('Київ', 'Львів'): 540,
    ('Київ', 'Одеса'): 475,
    ('Київ', 'Харків'): 480,
    ('Київ', 'Дніпро'): 475,
    ('Львів', 'Одеса'): 790,
    ('Львів', 'Харків'): 1050,
    ('Львів', 'Дніпро'): 930,
    ('Одеса', 'Харків'): 730,
    ('Одеса', 'Дніпро'): 500,
    ('Харків', 'Дніпро'): 210,
}


demo_tsp(distances=distances, start_city='Київ')

# %% [markdown]
# ## other:

# %%

# demo_comprehensive_comparison()


# %%
# main()

# %%



