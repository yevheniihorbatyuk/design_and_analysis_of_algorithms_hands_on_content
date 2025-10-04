# %% [markdown]
#  # Комплексний курс з алгоритмів та структур даних на графах
#
#  ## Розділ 1: Класичні алгоритми на графах
#
#  ### 🎯 Цілі розділу:
#  1. **Опанувати** фундаментальні алгоритми для пошуку шляхів, обходу та аналізу структури графів.
#  2. **Навчитись** застосовувати ці алгоритми для вирішення практичних задач з різних доменів.
#  3. **Розуміти** компроміси (швидкість, пам'ять, обмеження) між різними підходами.

# %%
# =============================================================================
# Клітинка 1: Налаштування середовища та імпорти
# =============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np

# Імпортуємо наші власні модулі
from src.algorithms.shortest_path import Dijkstra, AStar, BellmanFord, FloydWarshall
from src.algorithms.traversal import BFS, DFS
from src.algorithms.mst import Kruskal, Prim
from src.utils.graph_generator import GraphGenerator
from src.utils.visualizer import PathVisualizer, TraversalVisualizer, MstVisualizer

# Налаштування візуалізації
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12
print("✅ Середовище готове до роботи.")

# %% [markdown]
#  ## 1.1 Алгоритми пошуку найкоротшого шляху
#
#  Пошук найкоротшого шляху — одна з найбільш фундаментальних задач в теорії графів з величезною кількістю практичних застосувань, від GPS-навігації до маршрутизації інтернет-трафіку. У цьому розділі ми розглянемо ключові алгоритми та їхні особливості.

# %% [markdown]
#  ### Алгоритм Дейкстри (Dijkstra's Algorithm)
#
#  **Ідея:** "Жадібний" алгоритм, який знаходить найкоротший шлях від однієї стартової вершини до всіх інших у графі з **невід'ємними** вагами ребер. На кожному кроці він обирає найближчу ще не відвідану вершину і оновлює відстані до її сусідів.
#
#  | Складність | Використання пам'яті | Обмеження |
#  | :--- | :--- | :--- |
#  | `O(E log V)` | `O(V)` | Не працює з від'ємними вагами |
#
#  **Домени застосування:**
#  - **Логістика:** Розрахунок оптимальних маршрутів доставки.
#  - **Мережі:** Маршрутизація пакетів за протоколом OSPF.
#  - **Біоінформатика:** Аналіз метаболічних шляхів.

# %%
# =============================================================================
# Клітинка 2: Концептуальний приклад
# =============================================================================
conceptual_graph = nx.Graph()
edges = [('A', 'B', 4), ('A', 'C', 2), ('B', 'C', 1), ('B', 'D', 5), ('C', 'D', 8), ('C', 'E', 10), ('D', 'E', 2)]
conceptual_graph.add_weighted_edges_from(edges)
start_node, end_node = 'A', 'E'

path_info = Dijkstra.find_path(conceptual_graph, start_node, end_node)

print(f"Концептуальний приклад: пошук шляху з '{start_node}' до '{end_node}'")
print(f"  -> Знайдений шлях: {' → '.join(path_info['path'])}")
print(f"  -> Загальна відстань: {path_info['distance']:.2f}")

fig = PathVisualizer.draw_path(
    conceptual_graph,
    path=path_info['path'],
    title=f"Найкоротший шлях від {start_node} до {end_node} (Дейкстра)"
)
plt.show()

# %% [markdown]
#  #### Практичний приклад: Оптимізація логістики
#  **Задача:** У нас є мережа складів та магазинів. Потрібно знайти найдешевший маршрут доставки товару з головного складу (хабу) до віддаленого магазину. Вага ребер — це вартість перевезення між пунктами.

# %%
# =============================================================================
# Клітинка 3: Практичний приклад з домену "Логістика"
# =============================================================================
logistics_graph, node_types, pos = GraphGenerator.create_logistics_network(n_hubs=3, n_warehouses=10, n_stores=30, seed=42)
hub_node = 0
store_node = list(node_types.keys())[-1]

logistics_path_info = Dijkstra.find_path(logistics_graph, hub_node, store_node)

print(f"Практичний приклад: маршрут з Хабу-{hub_node} до Магазину-{store_node}")
print(f"  -> Оптимальний маршрут: {' → '.join(map(str, logistics_path_info['path']))}")
print(f"  -> Вартість доставки: ${logistics_path_info['distance']:.2f}")

fig = PathVisualizer.draw_logistics_network(
    logistics_graph, node_types, pos,
    path=logistics_path_info['path'],
    title="Оптимальний маршрут доставки в логістичній мережі"
)
plt.savefig('./data/dijkstra_logistics_path.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
#  ### Алгоритм A*
#
#  **Ідея:** Покращена версія Дейкстри, яка використовує **евристичну функцію** `h(n)` для оцінки відстані від поточної вершини `n` до цілі. Це дозволяє A* рухатися більш цілеспрямовано, замість того, щоб досліджувати граф у всіх напрямках. Пріоритет вершини визначається як `f(n) = g(n) + h(n)`, де `g(n)` — вже пройдений шлях.
#
#  **Ключова вимога:** Евристика має бути *допустимою* (admissible), тобто ніколи не переоцінювати реальну вартість шляху до цілі.

# %%
# =============================================================================
# Клітинка 4: Порівняльний аналіз Dijkstra vs A*
# =============================================================================
city_graph, pos = GraphGenerator.create_city_road_network(n_intersections=150, seed=101)
start_city, end_city = 0, 149

# Запускаємо обидва алгоритми
result_dijkstra = Dijkstra.find_path(city_graph, start_city, end_city)
result_astar = AStar.find_path(city_graph, start_city, end_city, positions=pos, heuristic='euclidean')

# Відображаємо результати у вигляді таблиці
comparison_df = pd.DataFrame([result_dijkstra, result_astar]).set_index('algorithm')
print("Порівняння продуктивності Dijkstra та A*:")
print(comparison_df[['distance', 'execution_time']].assign(visited_nodes=lambda df: [len(result_dijkstra['visited_nodes']), len(result_astar['visited_nodes'])]))

# Візуалізуємо відвідані вузли
fig = PathVisualizer.draw_visited_nodes_comparison(
    city_graph, pos,
    results=[result_dijkstra, result_astar],
    title="Порівняння досліджених зон: Dijkstra vs A*"
)
plt.savefig('./data/dijkstra_vs_astar_visited.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
#  #### Висновки з порівняння
#  На візуалізації чітко видно, що **Dijkstra** (зліва) досліджує велике коло навколо стартової точки, тоді як **A*** (справа) рухається вузьким "коридором" у напрямку цілі. Це призводить до значного зменшення кількості відвіданих вузлів та, як наслідок, до прискорення роботи, що критично важливо для великих карт.

# %% [markdown]
#  ### Алгоритм Беллмана-Форда (Bellman-Ford)
#
#  **Ідея:** На відміну від Дейкстри, цей алгоритм може працювати з графами, що містять ребра з **від'ємною вагою**. Він працює шляхом ітеративного оновлення (релаксації) відстаней до всіх вершин `V-1` разів. На `V`-й ітерації він може виявити наявність циклу з від'ємною сумарною вагою.
#
#  **Домени застосування:**
#  - **Фінанси:** Пошук арбітражних можливостей на валютних ринках (де від'ємний цикл означає безпрограшну послідовність обмінів).
#  - **Мережі:** Маршрутизація в мережах, де можуть бути "штрафи" (від'ємні ваги).

# %%
# =============================================================================
# Клітинка 5: Практичний приклад Bellman-Ford - Пошук арбітражу
# =============================================================================
# Створюємо граф валют. Вага log(rate) -> шлях це добуток курсів. Від'ємний цикл = прибуток
currencies = ['USD', 'EUR', 'GBP', 'JPY']
rates = [
    ('USD', 'EUR', 0.92), ('EUR', 'USD', 1.08),
    ('USD', 'GBP', 0.79), ('GBP', 'USD', 1.26),
    ('EUR', 'GBP', 0.85), ('GBP', 'EUR', 1.17),
    ('JPY', 'USD', 0.0067), ('USD', 'JPY', 149.0),
    # Створюємо арбітражну можливість
    ('EUR', 'JPY', 163.0), ('JPY', 'GBP', 0.0055), ('GBP', 'EUR', 1.18) # EUR->JPY->GBP->EUR
]
arbitrage_graph = nx.DiGraph()
for u, v, rate in rates:
    arbitrage_graph.add_edge(u, v, weight=-np.log(rate))

# Запускаємо Беллмана-Форда
# Для виявлення циклу, нам потрібно запустити його на всьому графі
# Для простоти, використаємо вбудовану функцію networkx, яка базується на Bellman-Ford
try:
    cycle = nx.find_negative_edge_cycle(arbitrage_graph, source='EUR')
    print("📈 Знайдено можливість для арбітражу (від'ємний цикл)!")
    print(f"   -> Шлях: {' → '.join(cycle)}")
except nx.NetworkXError:
    print("📉 Можливостей для арбітражу не знайдено.")

# %% [markdown]
#  ## 1.2 Алгоритми обходу графів
#
#  Обхід графу — це процес систематичного відвідування кожної вершини. Це фундаментальна операція, що лежить в основі багатьох складніших алгоритмів.

# %% [markdown]
#  ### Пошук в ширину (BFS - Breadth-First Search)
#
#  **Ідея:** BFS досліджує граф "рівнями", гарантуючи знаходження найкоротшого шляху в термінах кількості ребер.
#
#  **Практичне застосування:**
#  - **Соціальні мережі:** пошук "друзів друзів" (2-й рівень зв'язків).
#  - **Веб-краулери:** індексація сторінок, починаючи з головної.

# %%
# =============================================================================
# Клітинка 6: Практичний приклад BFS - Аналіз соціальної мережі
# =============================================================================
social_graph, names = GraphGenerator.create_social_network(seed=42)
start_user_id = 0
start_user_name = names[start_user_id]

bfs_traversal = BFS(social_graph)
result = bfs_traversal.traverse(start_node=start_user_id)

levels_df = pd.DataFrame(result['distances'].items(), columns=['User ID', 'Level']).sort_values(by='Level')
levels_df['Name'] = levels_df['User ID'].map(names)
print(f"Рівні зв'язків для користувача '{start_user_name}':")
print(levels_df[['Level', 'User ID', 'Name']].to_string(index=False))

fig = TraversalVisualizer.draw_bfs_levels(
    social_graph,
    start_node=start_user_id,
    distances=result['distances'],
    labels=names,
    title=f"Рівні зв'язків для '{start_user_name}' (BFS)"
)
plt.savefig('./data/bfs_social_network.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
#  ### Пошук в глибину (DFS - Depth-First Search)
#
#  **Ідея:** DFS йде "вглиб" по одній гілці графу до упору, перш ніж повернутися і дослідити інші шляхи.
#
#  **Практичне застосування:**
#  - **Виявлення циклів:** критично для перевірки залежностей (наприклад, при збірці ПЗ).
#  - **Топологічне сортування:** визначення правильного порядку виконання задач.
#  - **Генерація лабіринтів.**

# %%
# =============================================================================
# Клітинка 7: Практичний приклад DFS - Виявлення циклічних залежностей
# =============================================================================
dep_graph, tasks = GraphGenerator.create_dependency_graph_with_cycle()

dfs_traversal = DFS(dep_graph)
cycle = dfs_traversal.find_cycle()

fig = TraversalVisualizer.draw_cycle(
    dep_graph,
    cycle=[tasks[n] for n in cycle] if cycle else [],
    labels=tasks,
    title="Виявлення циклічних залежностей за допомогою DFS"
)
if cycle:
    print(f"🔥 Виявлено цикл: {' → '.join([tasks[n] for n in cycle])}")
else:
    print("✅ Циклічних залежностей не виявлено.")
plt.savefig('./data/dependency_cycle.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
#  ## 1.3 Мінімальне остівне дерево (MST)
#
#  **Задача:** З'єднати всі вершини графу разом за допомогою підмножини ребер, що має мінімальну можливу загальну вагу і не містить циклів.
#
#  **Домени застосування:**
#  - **Проектування мереж:** Дизайн мереж комунікацій (інтернет, телефон), електромереж, трубопроводів з мінімальними витратами.
#  - **Кластерний аналіз:** Використовується як крок в деяких алгоритмах кластеризації.
#  - **Комп'ютерне бачення:** Сегментація зображень.

# %% [markdown]
#  ### Алгоритм Краскала (Kruskal) vs Алгоритм Пріма (Prim)
#
#  | Алгоритм | Ідея | Складність | Найкраще для |
#  | :--- | :--- | :--- | :--- |
#  | **Kruskal**| "Лісовий" підхід: сортує всі ребра і додає найлегші, що не створюють циклів. | `O(E log E)` | Розріджені графи. |
#  | **Prim** | "Деревний" підхід: вирощує одне дерево, на кожному кроці додаючи найлегше ребро до нової вершини. | `O(E log V)` | Щільні графи. |

# %%
# =============================================================================
# Клітинка 8: Практичний приклад MST - Проектування мережі
# =============================================================================
# Генеруємо граф, що представляє міста та вартість прокладання кабелю
network_graph, pos = GraphGenerator.create_city_road_network(n_intersections=15, connectivity=0.5, seed=50)

# Знаходимо MST за допомогою обох алгоритмів
mst_kruskal = Kruskal.find_mst(network_graph)
mst_prim = Prim.find_mst(network_graph)

print("Порівняння алгоритмів MST:")
mst_comparison_data = {
    'Algorithm': ['Kruskal', 'Prim'],
    'Total Weight': [mst_kruskal['total_weight'], mst_prim['total_weight']],
    'Num Edges': [len(mst_kruskal['edges']), len(mst_prim['edges'])]
}
print(pd.DataFrame(mst_comparison_data))

# Візуалізуємо результат
fig = MstVisualizer.draw_mst(
    network_graph,
    mst_kruskal['edges'],
    title=f"Оптимальна мережа (MST) - Загальна вартість: {mst_kruskal['total_weight']:.2f}",
    pos=pos
)
plt.savefig('./data/mst_network_design.png', dpi=300, bbox_inches='tight')
plt.show()


# %% [markdown]
#  ## Завершення Розділу 1
#
#  Ми розглянули ключові класичні алгоритми для роботи з графами. Кожен з них має свою нішу застосування, і розуміння їхніх сильних та слабких сторін є критичним для ефективного вирішення реальних задач.
#
#  **Наступний розділ** буде присвячено структурам даних, які лежать в основі цих та багатьох інших алгоритмів.
# %%
