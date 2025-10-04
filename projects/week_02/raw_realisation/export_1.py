# %% [markdown]
#  # Алгоритми пошуку шляхів у графах
# 
#  ## Комплексний аналіз для навчального курсу
# 
# 
# 
#  ### 📚 Теоретичні основи
# 
# 
# 
#  Алгоритми пошуку найкоротшого шляху — фундаментальна область теорії графів
# 
#  з широким практичним застосуванням:
# 
# 
# 
#  1. **Навігаційні системи** (Google Maps, Waze, OSRM)
# 
#  2. **Соціальні мережі** (пошук зв'язків, рекомендації)
# 
#  3. **Логістика** (маршрутизація транспорту)
# 
#  4. **Ігрова індустрія** (пошук шляху NPC)
# 
#  5. **Мережеві протоколи** (маршрутизація пакетів)
# 
# 
# 
#  #### 🎯 Ключові поняття:
# 
#  - **Граф G = (V, E)**: V - вершини (вузли), E - ребра (зв'язки)
# 
#  - **Вага ребра**: числове значення "вартості" переходу
# 
#  - **Шлях**: послідовність вершин, з'єднаних ребрами
# 
#  - **Найкоротший шлях**: шлях з мінімальною сумарною вагою

# %%
# Імпорт необхідних бібліотек
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import heapq
import time
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Set, Optional, Callable
import seaborn as sns
import pandas as pd
from dataclasses import dataclass, field
import math
import warnings
warnings.filterwarnings('ignore')

# Налаштування візуалізації
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

print("🗺️ Лабораторія алгоритмів пошуку шляхів у графах")
print("=" * 80)


# %%
# Допоміжні класи та структури даних

@dataclass
class PathResult:
    """Результат пошуку шляху"""
    path: List[int]
    distance: float
    visited_nodes: int
    execution_time: float
    algorithm: str
    
@dataclass
class GraphStats:
    """Статистика графу"""
    nodes: int
    edges: int
    avg_degree: float
    density: float
    is_weighted: bool
    has_negative_weights: bool

class PriorityQueue:
    """Пріоритетна черга для алгоритмів пошуку"""
    def __init__(self):
        self.elements = []
        self.counter = 0
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, self.counter, item))
        self.counter += 1
    
    def get(self):
        return heapq.heappop(self.elements)[2]

class PerformanceTracker:
    """Трекер продуктивності алгоритмів"""
    def __init__(self):
        self.visited_count = 0
        self.comparisons = 0
        
    def reset(self):
        self.visited_count = 0
        self.comparisons = 0
    
    def visit_node(self):
        self.visited_count += 1
    
    def compare(self):
        self.comparisons += 1

tracker = PerformanceTracker()


# %%
# Генератори графів для різних доменів

class GraphGenerator:
    """Генератор графів для різних сценаріїв"""
    
    @staticmethod
    def create_city_road_network(n_intersections=50, connectivity=0.15, seed=42):
        """
        Створює граф міської дорожньої мережі
        Домен: Логістика, GPS-навігація
        """
        np.random.seed(seed)
        G = nx.Graph()
        
        # Розміщуємо перехрестя на сітці з невеликим шумом
        positions = {}
        grid_size = int(np.sqrt(n_intersections)) + 1
        for i in range(n_intersections):
            x = (i % grid_size) + np.random.uniform(-0.3, 0.3)
            y = (i // grid_size) + np.random.uniform(-0.3, 0.3)
            positions[i] = (x, y)
            G.add_node(i, pos=(x, y))
        
        # Додаємо ребра з вагами на основі евклідової відстані
        for i in range(n_intersections):
            for j in range(i + 1, n_intersections):
                if np.random.random() < connectivity:
                    dist = np.sqrt(
                        (positions[i][0] - positions[j][0])**2 + 
                        (positions[i][1] - positions[j][1])**2
                    )
                    # Вага = відстань + випадковий трафік
                    weight = dist * (1 + np.random.uniform(0, 0.5))
                    G.add_edge(i, j, weight=weight)
        
        return G, positions
    
    @staticmethod
    def create_social_network(n_users=100, avg_friends=8, seed=42):
        """
        Створює граф соціальної мережі
        Домен: Соціальні мережі (Facebook, LinkedIn)
        """
        np.random.seed(seed)
        # Використовуємо модель Watts-Strogatz для реалістичної соціальної мережі
        k = avg_friends
        p = 0.3  # ймовірність перез'єднання (створює "мости" між кластерами)
        G = nx.watts_strogatz_graph(n_users, k, p, seed=seed)
        
        # Додаємо ваги як "силу зв'язку" (кількість спільних інтересів)
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.randint(1, 10)
        
        return G
    
    @staticmethod
    def create_protein_network(n_proteins=80, interaction_prob=0.1, seed=42):
        """
        Створює мережу білок-білкових взаємодій
        Домен: Біоінформатика
        """
        np.random.seed(seed)
        G = nx.erdos_renyi_graph(n_proteins, interaction_prob, seed=seed)
        
        # Ваги як "сила взаємодії" або "енергія зв'язування"
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.uniform(0.1, 5.0)
        
        return G
    
    @staticmethod
    def create_game_dungeon(width=20, height=20, obstacle_prob=0.3, seed=42):
        """
        Створює підземелля для ігор
        Домен: Ігрова індустрія, роботехніка
        """
        np.random.seed(seed)
        G = nx.grid_2d_graph(width, height)
        
        # Перетворюємо на звичайний граф з числовими вершинами
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        
        # Видаляємо випадкові вершини (перешкоди)
        nodes_to_remove = [n for n in G.nodes() if np.random.random() < obstacle_prob]
        G.remove_nodes_from(nodes_to_remove)
        
        # Додаємо ваги (складність терену)
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.choice([1, 2, 3, 5], p=[0.5, 0.3, 0.15, 0.05])
        
        # Зберігаємо сітку для візуалізації
        positions = {i: (i % width, i // width) for i in G.nodes()}
        
        return G, positions
    
    @staticmethod
    def create_network_topology(n_routers=60, seed=42):
        """
        Створює топологію комп'ютерної мережі
        Домен: Мережеві протоколи, маршрутизація
        """
        np.random.seed(seed)
        # Використовуємо модель Barabási-Albert для scale-free мережі
        G = nx.barabasi_albert_graph(n_routers, 3, seed=seed)
        
        # Ваги як затримка (latency) у мс
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.uniform(1, 50)
        
        return G
    
    @staticmethod
    def create_logistics_network(n_warehouses=30, n_stores=40, seed=42):
        """
        Створює логістичну мережу доставки
        Домен: Логістика, supply chain
        """
        np.random.seed(seed)
        G = nx.Graph()
        
        # Додаємо склади та магазини
        for i in range(n_warehouses):
            G.add_node(i, node_type='warehouse')
        
        for i in range(n_warehouses, n_warehouses + n_stores):
            G.add_node(i, node_type='store')
        
        # Склади з'єднані між собою
        for i in range(n_warehouses):
            for j in range(i + 1, n_warehouses):
                if np.random.random() < 0.4:
                    G.add_edge(i, j, weight=np.random.uniform(50, 200))
        
        # Кожен магазин з'єднаний з кількома складами
        for store in range(n_warehouses, n_warehouses + n_stores):
            n_connections = np.random.randint(2, 5)
            warehouses = np.random.choice(n_warehouses, n_connections, replace=False)
            for wh in warehouses:
                G.add_edge(store, wh, weight=np.random.uniform(20, 100))
        
        return G
    
    @staticmethod
    def create_flight_network(n_airports=50, hub_count=5, seed=42):
        """
        Створює мережу авіарейсів
        Домен: Логістика, авіаперевезення
        """
        np.random.seed(seed)
        G = nx.Graph()
        
        # Визначаємо хаби (великі аеропорти)
        hubs = list(range(hub_count))
        regular = list(range(hub_count, n_airports))
        
        # Хаби з'єднані між собою
        for i in hubs:
            G.add_node(i, hub=True)
            for j in hubs:
                if i < j:
                    G.add_edge(i, j, weight=np.random.uniform(100, 500))
        
        # Регулярні аеропорти
        for i in regular:
            G.add_node(i, hub=False)
            # Кожен з'єднаний з 2-4 хабами
            connected_hubs = np.random.choice(hubs, np.random.randint(2, 5), replace=False)
            for hub in connected_hubs:
                G.add_edge(i, hub, weight=np.random.uniform(50, 300))
            
            # Деякі прямі рейси між регулярними
            if np.random.random() < 0.1:
                other = np.random.choice([x for x in regular if x != i])
                if not G.has_edge(i, other):
                    G.add_edge(i, other, weight=np.random.uniform(100, 400))
        
        return G


# %% [markdown]
#  ## 1. Dijkstra's Algorithm
# 
# 
# 
#  ### Опис:
# 
#  Класичний алгоритм пошуку найкоротшого шляху від однієї вершини до всіх інших
# 
#  у графі з невід'ємними вагами ребер.
# 
# 
# 
#  ### Складність:
# 
#  - **Часова**: O(E log V) з бінарною купою, O(E + V log V) з купою Фібоначчі
# 
#  - **Просторова**: O(V)
# 
# 
# 
#  ### Переваги:
# 
#  - Гарантує оптимальний шлях
# 
#  - Ефективний для графів без від'ємних ваг
# 
#  - Можна зупинити при досягненні цільової вершини
# 
# 
# 
#  ### Недоліки:
# 
#  - Не працює з від'ємними вагами
# 
#  - Досліджує багато вершин у всіх напрямках

# %%
class DijkstraAlgorithm:
    """Реалізація алгоритму Дейкстри"""
    
    @staticmethod
    def find_shortest_path(graph: nx.Graph, start: int, goal: int) -> PathResult:
        """
        Знаходить найкоротший шлях від start до goal
        """
        start_time = time.perf_counter()
        tracker.reset()
        
        # Ініціалізація
        distances = {node: float('inf') for node in graph.nodes()}
        distances[start] = 0
        previous = {node: None for node in graph.nodes()}
        pq = PriorityQueue()
        pq.put(start, 0)
        visited = set()
        
        while not pq.empty():
            current = pq.get()
            
            if current == goal:
                break
            
            if current in visited:
                continue
            
            visited.add(current)
            tracker.visit_node()
            
            # Перевіряємо всіх сусідів
            for neighbor in graph.neighbors(current):
                tracker.compare()
                weight = graph[current][neighbor].get('weight', 1)
                distance = distances[current] + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
                    pq.put(neighbor, distance)
        
        # Відновлення шляху
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        return PathResult(
            path=path if path[0] == start else [],
            distance=distances[goal] if distances[goal] != float('inf') else -1,
            visited_nodes=tracker.visited_count,
            execution_time=execution_time,
            algorithm="Dijkstra"
        )
    
    @staticmethod
    def find_all_shortest_paths(graph: nx.Graph, start: int) -> Dict[int, float]:
        """
        Знаходить найкоротші шляхи від start до всіх інших вершин
        """
        distances = {node: float('inf') for node in graph.nodes()}
        distances[start] = 0
        pq = PriorityQueue()
        pq.put(start, 0)
        visited = set()
        
        while not pq.empty():
            current = pq.get()
            
            if current in visited:
                continue
            
            visited.add(current)
            
            for neighbor in graph.neighbors(current):
                weight = graph[current][neighbor].get('weight', 1)
                distance = distances[current] + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    pq.put(neighbor, distance)
        
        return distances


# %%
def test_dijkstra_scenarios():
    """Тестування Dijkstra на різних сценаріях"""
    print("\n🔍 ТЕСТУВАННЯ АЛГОРИТМУ ДЕЙКСТРИ")
    print("=" * 80)
    
    scenarios = [
        ("Міська дорожня мережа", *GraphGenerator.create_city_road_network(50)),
        ("Соціальна мережа", GraphGenerator.create_social_network(100), None),
        ("Мережа авіарейсів", GraphGenerator.create_flight_network(50), None),
        ("Логістична мережа", GraphGenerator.create_logistics_network(30, 40), None),
    ]
    
    results = []
    
    for name, graph, positions in scenarios:
        print(f"\n📊 Сценарій: {name}")
        print(f"   Вершин: {graph.number_of_nodes()}, Ребер: {graph.number_of_edges()}")
        
        # Вибираємо випадкові start і goal
        nodes = list(graph.nodes())
        start, goal = np.random.choice(nodes, 2, replace=False)
        
        result = DijkstraAlgorithm.find_shortest_path(graph, start, goal)
        
        print(f"   Шлях: {start} → {goal}")
        print(f"   Відстань: {result.distance:.2f}")
        print(f"   Відвідано вузлів: {result.visited_nodes}")
        print(f"   Час виконання: {result.execution_time:.3f}ms")
        print(f"   Довжина шляху: {len(result.path)} вузлів")
        
        results.append({
            'Scenario': name,
            'Nodes': graph.number_of_nodes(),
            'Edges': graph.number_of_edges(),
            'Distance': result.distance,
            'Visited': result.visited_nodes,
            'Time (ms)': result.execution_time
        })
    
    return pd.DataFrame(results)

dijkstra_results = test_dijkstra_scenarios()


# %% [markdown]
#  ## 2. A* Algorithm
# 
# 
# 
#  ### Опис:
# 
#  Евристичний алгоритм пошуку, що використовує функцію оцінки f(n) = g(n) + h(n),
# 
#  де g(n) - фактична вартість шляху, h(n) - евристична оцінка до цілі.
# 
# 
# 
#  ### Складність:
# 
#  - **Часова**: O(E) у середньому (залежить від евристики)
# 
#  - **Просторова**: O(V)
# 
# 
# 
#  ### Переваги:
# 
#  - Значно швидше ніж Dijkstra при хорошій евристиці
# 
#  - Оптимальний при допустимій евристиці (h(n) ≤ реальна відстань)
# 
#  - Ідеально для просторового пошуку
# 
# 
# 
#  ### Недоліки:
# 
#  - Потребує евристичної функції
# 
#  - Якість залежить від якості евристики

# %%
class AStarAlgorithm:
    """Реалізація алгоритму A*"""
    
    @staticmethod
    def manhattan_distance(pos1, pos2):
        """Манхеттенська відстань (для сіток)"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    @staticmethod
    def euclidean_distance(pos1, pos2):
        """Евклідова відстань (для географічних координат)"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    @staticmethod
    def find_shortest_path(graph: nx.Graph, start: int, goal: int, 
                          positions: Dict = None, heuristic: str = 'euclidean') -> PathResult:
        """
        Знаходить найкоротший шлях використовуючи A*
        """
        start_time = time.perf_counter()
        tracker.reset()
        
        # Вибір евристики
        if positions is None:
            # Якщо немає координат, використовуємо нульову евристику (A* = Dijkstra)
            heuristic_func = lambda n: 0
        else:
            if heuristic == 'manhattan':
                heuristic_func = lambda n: AStarAlgorithm.manhattan_distance(
                    positions.get(n, (0, 0)), positions.get(goal, (0, 0))
                )
            else:  # euclidean
                heuristic_func = lambda n: AStarAlgorithm.euclidean_distance(
                    positions.get(n, (0, 0)), positions.get(goal, (0, 0))
                )
        
        # Ініціалізація
        g_score = {node: float('inf') for node in graph.nodes()}
        g_score[start] = 0
        
        f_score = {node: float('inf') for node in graph.nodes()}
        f_score[start] = heuristic_func(start)
        
        previous = {node: None for node in graph.nodes()}
        pq = PriorityQueue()
        pq.put(start, f_score[start])
        visited = set()
        
        while not pq.empty():
            current = pq.get()
            
            if current == goal:
                break
            
            if current in visited:
                continue
            
            visited.add(current)
            tracker.visit_node()
            
            for neighbor in graph.neighbors(current):
                tracker.compare()
                weight = graph[current][neighbor].get('weight', 1)
                tentative_g_score = g_score[current] + weight
                
                if tentative_g_score < g_score[neighbor]:
                    previous[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic_func(neighbor)
                    pq.put(neighbor, f_score[neighbor])
        
        # Відновлення шляху
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        return PathResult(
            path=path if path and path[0] == start else [],
            distance=g_score[goal] if g_score[goal] != float('inf') else -1,
            visited_nodes=tracker.visited_count,
            execution_time=execution_time,
            algorithm="A*"
        )


# %%
def test_astar_vs_dijkstra():
    """Порівняння A* з Dijkstra"""
    print("\n⚡ ПОРІВНЯННЯ A* vs DIJKSTRA")
    print("=" * 80)
    
    # Створюємо граф міста з координатами
    graph, positions = GraphGenerator.create_city_road_network(100, 0.12)
    
    nodes = list(graph.nodes())
    start, goal = nodes[0], nodes[-1]
    
    print(f"\nГраф: {graph.number_of_nodes()} вузлів, {graph.number_of_edges()} ребер")
    print(f"Маршрут: {start} → {goal}")
    print(f"Пряма відстань: {AStarAlgorithm.euclidean_distance(positions[start], positions[goal]):.2f}")
    
    # Dijkstra
    result_dijkstra = DijkstraAlgorithm.find_shortest_path(graph, start, goal)
    
    # A* з евклідовою евристикою
    result_astar_euclidean = AStarAlgorithm.find_shortest_path(
        graph, start, goal, positions, 'euclidean'
    )
    
    # A* з манхеттенською евристикою
    result_astar_manhattan = AStarAlgorithm.find_shortest_path(
        graph, start, goal, positions, 'manhattan'
    )
    
    comparison = pd.DataFrame([
        {
            'Algorithm': 'Dijkstra',
            'Distance': result_dijkstra.distance,
            'Visited Nodes': result_dijkstra.visited_nodes,
            'Time (ms)': result_dijkstra.execution_time,
            'Path Length': len(result_dijkstra.path)
        },
        {
            'Algorithm': 'A* (Euclidean)',
            'Distance': result_astar_euclidean.distance,
            'Visited Nodes': result_astar_euclidean.visited_nodes,
            'Time (ms)': result_astar_euclidean.execution_time,
            'Path Length': len(result_astar_euclidean.path)
        },
        {
            'Algorithm': 'A* (Manhattan)',
            'Distance': result_astar_manhattan.distance,
            'Visited Nodes': result_astar_manhattan.visited_nodes,
            'Time (ms)': result_astar_manhattan.execution_time,
            'Path Length': len(result_astar_manhattan.path)
        }
    ])
    
    print("\n" + comparison.to_string(index=False))
    
    # Розраховуємо прискорення
    speedup_euclidean = result_dijkstra.visited_nodes / result_astar_euclidean.visited_nodes
    speedup_manhattan = result_dijkstra.visited_nodes / result_astar_manhattan.visited_nodes
    
    print(f"\n📈 Прискорення A* (Euclidean): {speedup_euclidean:.2f}x менше вузлів")
    print(f"📈 Прискорення A* (Manhattan): {speedup_manhattan:.2f}x менше вузлів")
    
    return comparison

astar_comparison = test_astar_vs_dijkstra()


# %% [markdown]
#  ## 3. Bellman-Ford Algorithm
# 
# 
# 
#  ### Опис:
# 
#  Алгоритм, що працює з від'ємними вагами та виявляє від'ємні цикли.
# 
#  Використовує динамічне програмування.
# 
# 
# 
#  ### Складність:
# 
#  - **Часова**: O(V·E)
# 
#  - **Просторова**: O(V)
# 
# 
# 
#  ### Переваги:
# 
#  - Працює з від'ємними вагами
# 
#  - Виявляє від'ємні цикли
# 
#  - Простий у реалізації
# 
# 
# 
#  ### Недоліки:
# 
#  - Повільніший за Dijkstra
# 
#  - Не підходить для великих графів

# %%
# class PathResult:
#     def __init__(self, path, distance, visited_nodes, execution_time, algorithm):
#         self.path = path
#         self.distance = distance
#         self.visited_nodes = visited_nodes
#         self.execution_time = execution_time
#         self.algorithm = algorithm

# class Tracker:
#     def __init__(self):
#         self.visited_count = 0
#         self.compare_count = 0

#     def reset(self):
#         self.visited_count = 0
#         self.compare_count = 0

#     def visit_node(self):
#         self.visited_count += 1

#     def compare(self):
#         self.compare_count += 1

# tracker = Tracker()

# class BellmanFordAlgorithm:
#     """Реалізація алгоритму Беллмана-Форда"""
    
#     @staticmethod
#     def find_shortest_path(graph: nx.Graph, start: int, goal: int) -> Tuple[PathResult, bool]:
#         """
#         Знаходить найкоротший шлях, працює з від'ємними вагами
#         Повертає (PathResult, has_negative_cycle)
#         """
#         start_time = time.perf_counter()
#         tracker.reset()
        
#         # Перетворюємо неорієнтований граф на орієнтований для Bellman-Ford
#         # Кожне ребро {u,v} стає двома ребрами (u,v) і (v,u)
#         directed_graph = nx.DiGraph()
#         for u, v, data in graph.edges(data=True):
#             weight = data.get('weight', 1)
#             directed_graph.add_edge(u, v, weight=weight)
#             directed_graph.add_edge(v, u, weight=weight)
        
#         nodes = list(directed_graph.nodes())
        
#         # Ініціалізація
#         distances = {node: float('inf') for node in nodes}
#         distances[start] = 0
#         previous = {node: None for node in nodes}
        
#         # Релаксація ребер V-1 разів
#         for _ in range(len(nodes) - 1):
#             tracker.visit_node()
#             for u, v in directed_graph.edges():
#                 tracker.compare()
#                 weight = directed_graph[u][v].get('weight', 1)
                
#                 if distances[u] != float('inf') and distances[u] + weight < distances[v]:
#                     distances[v] = distances[u] + weight
#                     previous[v] = u
        
#         # Перевірка на від'ємні цикли
#         has_negative_cycle = False
#         for u, v in directed_graph.edges():
#             weight = directed_graph[u][v].get('weight', 1)
#             if distances[u] != float('inf') and distances[u] + weight < distances[v]:
#                 has_negative_cycle = True
#                 break
        
#         # Відновлення шляху
#         path = []
#         current = goal
#         while current is not None:
#             path.append(current)
#             current = previous[current]
#         path.reverse()
        
#         execution_time = (time.perf_counter() - start_time) * 1000
        
#         return PathResult(
#             path=path if path and path[0] == start else [],
#             distance=distances[goal] if distances[goal] != float('inf') else -1,
#             visited_nodes=tracker.visited_count,
#             execution_time=execution_time,
#             algorithm="Bellman-Ford"
#         ), has_negative_cycle

# %%
# def test_bellman_ford_with_negative_weights():
#     """Тестування Bellman-Ford з від'ємними вагами"""
#     print("\n⚖️ ТЕСТУВАННЯ BELLMAN-FORD З ВІД'ЄМНИМИ ВАГАМИ")
#     print("=" * 80)
    
#     # Створюємо граф з від'ємними вагами (наприклад, арбітраж валют)
#     G = nx.Graph()
    
#     # Додаємо валютні пари з "прибутком/витратами"
#     currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF']
#     currency_map = {curr: i for i, curr in enumerate(currencies)}
    
#     # Додаємо ребра (обмінні курси з можливим прибутком/збитком)
#     exchanges = [
#         ('USD', 'EUR', -0.02),  # Невелика комісія
#         ('EUR', 'GBP', 0.05),   # Прибуток
#         ('GBP', 'JPY', -0.03),
#         ('JPY', 'CHF', 0.02),
#         ('CHF', 'USD', -0.01),
#         ('USD', 'GBP', -0.04),
#         ('EUR', 'JPY', 0.01),
#         ('GBP', 'CHF', -0.02),
#     ]
    
#     for curr1, curr2, rate in exchanges:
#         G.add_edge(currency_map[curr1], currency_map[curr2], weight=rate)
    
#     print("\nСценарій: Арбітраж валют (від'ємні ваги = прибуток)")
#     print(f"Валюти: {currencies}")
#     print(f"Обмінних пар: {G.number_of_edges()}")
    
#     start = currency_map['USD']
#     goal = currency_map['EUR']
    
#     result, has_negative_cycle = BellmanFordAlgorithm.find_shortest_path(G, start, goal)
    
#     print(f"\nМаршрут: USD → EUR")
#     print(f"Оптимальна вартість: {result.distance:.4f}")
#     print(f"Шлях: {' → '.join([currencies[i] for i in result.path])}")
#     print(f"Виявлено від'ємний цикл: {'Так (можливий арбітраж!)' if has_negative_cycle else 'Ні'}")
#     print(f"Час виконання: {result.execution_time:.3f}ms")
    
#     return result, has_negative_cycle

# bellman_ford_result = test_bellman_ford_with_negative_weights()


# %% [markdown]
#  ## 4. Floyd-Warshall Algorithm
# 
# 
# 
#  ### Опис:
# 
#  Знаходить найкоротші шляхи між усіма парами вершин.
# 
#  Використовує динамічне програмування.
# 
# 
# 
#  ### Складність:
# 
#  - **Часова**: O(V³)
# 
#  - **Просторова**: O(V²)
# 
# 
# 
#  ### Переваги:
# 
#  - Знаходить всі шляхи за один прохід
# 
#  - Працює з від'ємними вагами
# 
#  - Проста реалізація
# 
# 
# 
#  ### Недоліки:
# 
#  - Кубічна складність
# 
#  - Великі вимоги до пам'яті
# 
#  - Не підходить для великих графів (>500 вузлів)

# %%
class FloydWarshallAlgorithm:
    """Реалізація алгоритму Флойда-Воршелла"""
    
    @staticmethod
    def find_all_shortest_paths(graph: nx.Graph) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Знаходить найкоротші шляхи між усіма парами вершин
        Повертає (матриця_відстаней, матриця_наступних_вершин, час_виконання)
        """
        start_time = time.perf_counter()
        
        nodes = list(graph.nodes())
        n = len(nodes)
        node_index = {node: i for i, node in enumerate(nodes)}
        
        # Ініціалізація матриць
        dist = np.full((n, n), np.inf)
        next_node = np.full((n, n), -1, dtype=int)
        
        # Відстань від вершини до себе = 0
        for i in range(n):
            dist[i][i] = 0
        
        # Заповнюємо початкові відстані з графу
        for u, v in graph.edges():
            i, j = node_index[u], node_index[v]
            weight = graph[u][v].get('weight', 1)
            dist[i][j] = weight
            dist[j][i] = weight  # Для неорієнтованого графу
            next_node[i][j] = j
            next_node[j][i] = i
        
        # Основний алгоритм Floyd-Warshall
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next_node[i][j] = next_node[i][k]
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        return dist, next_node, execution_time, node_index
    
    @staticmethod
    def reconstruct_path(next_node: np.ndarray, node_index: Dict, start: int, goal: int) -> List[int]:
        """Відновлення шляху з матриці next_node"""
        if next_node[node_index[start]][node_index[goal]] == -1:
            return []
        
        path = [start]
        current = start
        while current != goal:
            current = next_node[node_index[current]][node_index[goal]]
            if current == -1:
                return []
            # Знаходимо оригінальний вузол з індексу
            current = [k for k, v in node_index.items() if v == current][0]
            path.append(current)
        
        return path


# %%
def test_floyd_warshall_applications():
    """Тестування Floyd-Warshall на різних застосуваннях"""
    print("\n🌐 ТЕСТУВАННЯ FLOYD-WARSHALL (ВСІ ПАРИ ШЛЯХІВ)")
    print("=" * 80)
    
    # Сценарій 1: Мала мережа доставки
    print("\n📦 Сценарій 1: Мережа доставки (всі відстані між точками)")
    G_delivery = GraphGenerator.create_logistics_network(n_warehouses=8, n_stores=12)
    
    dist, next_node, exec_time, node_index = FloydWarshallAlgorithm.find_all_shortest_paths(G_delivery)
    
    print(f"Вузлів: {len(G_delivery.nodes())}")
    print(f"Час виконання: {exec_time:.2f}ms")
    print(f"Обчислено {len(G_delivery.nodes())**2} пар відстаней")
    
    # Знаходимо найдовшу найкоротшу відстань (діаметр графу)
    max_dist = np.max(dist[dist != np.inf])
    print(f"Діаметр мережі (найдовша відстань): {max_dist:.2f}")
    
    # Знаходимо центральний вузол (з мінімальною максимальною відстанню)
    max_distances = np.max(dist, axis=1)
    max_distances[max_distances == np.inf] = np.nan
    central_node_idx = np.nanargmin(max_distances)
    central_node = [k for k, v in node_index.items() if v == central_node_idx][0]
    print(f"Центральний вузол (оптимальний для розміщення): {central_node}")
    
    # Сценарій 2: Соціальна мережа - ступені розділення
    print("\n👥 Сценарій 2: Соціальна мережа (ступені розділення)")
    G_social = GraphGenerator.create_social_network(n_users=50, avg_friends=6)
    
    dist_social, _, exec_time_social, node_index_social = FloydWarshallAlgorithm.find_all_shortest_paths(G_social)
    
    # Підраховуємо розподіл ступенів розділення
    separation_degrees = dist_social[dist_social != np.inf]
    separation_degrees = separation_degrees[separation_degrees != 0]
    
    print(f"Користувачів: {len(G_social.nodes())}")
    print(f"Час виконання: {exec_time_social:.2f}ms")
    print(f"Середній ступінь розділення: {np.mean(separation_degrees):.2f}")
    print(f"Максимальний ступінь розділення: {int(np.max(separation_degrees))}")
    
    # Розподіл ступенів
    unique, counts = np.unique(separation_degrees.astype(int), return_counts=True)
    print("\nРозподіл ступенів розділення:")
    for degree, count in zip(unique, counts):
        print(f"  {degree} ступінь: {count} пар ({count/len(separation_degrees)*100:.1f}%)")
    
    return dist, dist_social

floyd_warshall_results = test_floyd_warshall_applications()


# %% [markdown]
#  ## 5. Bidirectional Search
# 
# 
# 
#  ### Опис:
# 
#  Виконує пошук одночасно від початкової та кінцевої вершин,
# 
#  зустрічаючись посередині.
# 
# 
# 
#  ### Складність:
# 
#  - **Часова**: O(b^(d/2)) де b - фактор розгалуження, d - глибина
# 
#  - **Просторова**: O(b^(d/2))
# 
# 
# 
#  ### Переваги:
# 
#  - Значно швидше за односпрямований пошук
# 
#  - Зменшує кількість відвіданих вузлів експоненційно
# 
# 
# 
#  ### Недоліки:
# 
#  - Потрібно знати кінцеву вершину
# 
#  - Складніша реалізація
# 
#  - Потребує більше пам'яті

# %%
class BidirectionalSearch:
    """Реалізація двонаправленого пошуку"""
    
    @staticmethod
    def find_shortest_path(graph: nx.Graph, start: int, goal: int) -> PathResult:
        """
        Двонаправлений пошук з використанням BFS
        """
        start_time = time.perf_counter()
        tracker.reset()
        
        if start == goal:
            return PathResult([start], 0, 1, 0, "Bidirectional")
        
        # Ініціалізація для пошуку з початку
        queue_start = deque([start])
        visited_start = {start: None}
        dist_start = {start: 0}
        
        # Ініціалізація для пошуку з кінця
        queue_goal = deque([goal])
        visited_goal = {goal: None}
        dist_goal = {goal: 0}
        
        # Точка зустрічі
        meeting_node = None
        
        while queue_start and queue_goal:
            # Крок з початку
            if queue_start:
                current_start = queue_start.popleft()
                tracker.visit_node()
                
                for neighbor in graph.neighbors(current_start):
                    if neighbor not in visited_start:
                        visited_start[neighbor] = current_start
                        weight = graph[current_start][neighbor].get('weight', 1)
                        dist_start[neighbor] = dist_start[current_start] + weight
                        queue_start.append(neighbor)
                        
                        # Перевірка на зустріч
                        if neighbor in visited_goal:
                            meeting_node = neighbor
                            break
                
                if meeting_node:
                    break
            
            # Крок з кінця
            if queue_goal:
                current_goal = queue_goal.popleft()
                tracker.visit_node()
                
                for neighbor in graph.neighbors(current_goal):
                    if neighbor not in visited_goal:
                        visited_goal[neighbor] = current_goal
                        weight = graph[current_goal][neighbor].get('weight', 1)
                        dist_goal[neighbor] = dist_goal[current_goal] + weight
                        queue_goal.append(neighbor)
                        
                        # Перевірка на зустріч
                        if neighbor in visited_start:
                            meeting_node = neighbor
                            break
                
                if meeting_node:
                    break
        
        if meeting_node is None:
            return PathResult([], -1, tracker.visited_count, 0, "Bidirectional")
        
        # Відновлення шляху
        path_start = []
        current = meeting_node
        while current is not None:
            path_start.append(current)
            current = visited_start[current]
        path_start.reverse()
        
        path_goal = []
        current = visited_goal[meeting_node]
        while current is not None:
            path_goal.append(current)
            current = visited_goal[current]
        
        path = path_start + path_goal
        total_distance = dist_start[meeting_node] + dist_goal[meeting_node]
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        return PathResult(
            path=path,
            distance=total_distance,
            visited_nodes=tracker.visited_count,
            execution_time=execution_time,
            algorithm="Bidirectional"
        )


# %%
def test_bidirectional_vs_dijkstra():
    """Порівняння двонаправленого пошуку з Dijkstra"""
    print("\n↔️ ПОРІВНЯННЯ BIDIRECTIONAL SEARCH vs DIJKSTRA")
    print("=" * 80)
    
    # Тестуємо на різних розмірах графів
    sizes = [50, 100, 200, 500]
    results = []
    
    for size in sizes:
        print(f"\n📊 Розмір графу: {size} вузлів")
        
        graph, positions = GraphGenerator.create_city_road_network(size, 0.1)
        nodes = list(graph.nodes())
        
        # Вибираємо віддалені вузли
        start = nodes[0]
        goal = nodes[-1]
        
        # Dijkstra
        result_dijkstra = DijkstraAlgorithm.find_shortest_path(graph, start, goal)
        
        # Bidirectional
        result_bidirectional = BidirectionalSearch.find_shortest_path(graph, start, goal)
        
        speedup = result_dijkstra.visited_nodes / result_bidirectional.visited_nodes if result_bidirectional.visited_nodes > 0 else 1
        
        print(f"  Dijkstra: {result_dijkstra.visited_nodes} вузлів, {result_dijkstra.execution_time:.3f}ms")
        print(f"  Bidirectional: {result_bidirectional.visited_nodes} вузлів, {result_bidirectional.execution_time:.3f}ms")
        print(f"  Прискорення: {speedup:.2f}x менше вузлів")
        
        results.append({
            'Size': size,
            'Dijkstra Visited': result_dijkstra.visited_nodes,
            'Bidirectional Visited': result_bidirectional.visited_nodes,
            'Speedup': speedup,
            'Dijkstra Time': result_dijkstra.execution_time,
            'Bidirectional Time': result_bidirectional.execution_time
        })
    
    return pd.DataFrame(results)

bidirectional_comparison = test_bidirectional_vs_dijkstra()


# %% [markdown]
#  ## 6. Contraction Hierarchies (CH)
# 
# 
# 
#  ### Опис:
# 
#  Метод попередньої обробки графу для надшвидкого пошуку.
# 
#  Створює ієрархію вузлів за важливістю.
# 
# 
# 
#  ### Складність:
# 
#  - **Препроцесинг**: O(n log n)
# 
#  - **Запит**: O(log n)
# 
#  - **Просторова**: O(E log V)
# 
# 
# 
#  ### Переваги:
# 
#  - Надзвичайно швидкі запити (мікросекунди)
# 
#  - Основа сучасних навігаційних систем
# 
#  - Масштабується на мільйони вузлів
# 
# 
# 
#  ### Недоліки:
# 
#  - Довгий препроцесинг
# 
#  - Багато додаткової пам'яті
# 
#  - Складна реалізація

# %%
class ContractionHierarchies:
    """Спрощена реалізація Contraction Hierarchies"""
    
    def __init__(self, graph: nx.Graph):
        self.original_graph = graph
        self.node_order = []
        self.shortcuts = {}
        self.levels = {}
        
    def preprocess(self) -> float:
        """
        Виконує препроцесинг графу
        Повертає час виконання
        """
        start_time = time.perf_counter()
        
        # Спрощений варіант: сортуємо вузли за ступенем
        nodes = list(self.original_graph.nodes())
        node_degrees = [(node, self.original_graph.degree(node)) for node in nodes]
        node_degrees.sort(key=lambda x: x[1])
        
        self.node_order = [node for node, _ in node_degrees]
        
        # Присвоюємо рівні
        for level, node in enumerate(self.node_order):
            self.levels[node] = level
        
        # Створюємо shortcuts (спрощено)
        working_graph = self.original_graph.copy()
        
        for level, node in enumerate(self.node_order[:-1]):  # Не обробляємо останній
            neighbors = list(working_graph.neighbors(node))
            
            # Для кожної пари сусідів перевіряємо, чи потрібен shortcut
            for i, u in enumerate(neighbors):
                for v in neighbors[i+1:]:
                    if not working_graph.has_edge(u, v):
                        # Перевіряємо, чи шлях через node коротший
                        dist_through_node = (
                            working_graph[u][node].get('weight', 1) +
                            working_graph[node][v].get('weight', 1)
                        )
                        
                        # Додаємо shortcut
                        if (u, v) not in self.shortcuts:
                            self.shortcuts[(u, v)] = dist_through_node
                            self.shortcuts[(v, u)] = dist_through_node
            
            # Видаляємо вузол з робочого графу
            working_graph.remove_node(node)
        
        preprocessing_time = (time.perf_counter() - start_time) * 1000
        return preprocessing_time
    
    def query(self, start: int, goal: int) -> PathResult:
        """
        Швидкий запит після препроцесингу
        """
        start_time = time.perf_counter()
        tracker.reset()
        
        # Спрощений bidirectional Dijkstra по ієрархії
        # Forward search (тільки вгору по ієрархії)
        dist_forward = {start: 0}
        pq_forward = PriorityQueue()
        pq_forward.put(start, 0)
        visited_forward = set()
        
        # Backward search (тільки вгору по ієрархії)
        dist_backward = {goal: 0}
        pq_backward = PriorityQueue()
        pq_backward.put(goal, 0)
        visited_backward = set()
        
        best_distance = float('inf')
        meeting_node = None
        
        # Forward search
        while not pq_forward.empty():
            current = pq_forward.get()
            if current in visited_forward:
                continue
            
            visited_forward.add(current)
            tracker.visit_node()
            
            # Тільки до вузлів вище в ієрархії
            for neighbor in self.original_graph.neighbors(current):
                if self.levels[neighbor] > self.levels[current]:
                    weight = self.original_graph[current][neighbor].get('weight', 1)
                    # Перевіряємо shortcuts
                    if (current, neighbor) in self.shortcuts:
                        weight = min(weight, self.shortcuts[(current, neighbor)])
                    
                    new_dist = dist_forward[current] + weight
                    if neighbor not in dist_forward or new_dist < dist_forward[neighbor]:
                        dist_forward[neighbor] = new_dist
                        pq_forward.put(neighbor, new_dist)
        
        # Backward search
        while not pq_backward.empty():
            current = pq_backward.get()
            if current in visited_backward:
                continue
            
            visited_backward.add(current)
            tracker.visit_node()
            
            for neighbor in self.original_graph.neighbors(current):
                if self.levels[neighbor] > self.levels[current]:
                    weight = self.original_graph[current][neighbor].get('weight', 1)
                    if (current, neighbor) in self.shortcuts:
                        weight = min(weight, self.shortcuts[(current, neighbor)])
                    
                    new_dist = dist_backward[current] + weight
                    if neighbor not in dist_backward or new_dist < dist_backward[neighbor]:
                        dist_backward[neighbor] = new_dist
                        pq_backward.put(neighbor, new_dist)
        
        # Знаходимо найкоротшу відстань через загальні вузли
        common_nodes = visited_forward & visited_backward
        for node in common_nodes:
            total_dist = dist_forward.get(node, float('inf')) + dist_backward.get(node, float('inf'))
            if total_dist < best_distance:
                best_distance = total_dist
                meeting_node = node
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        return PathResult(
            path=[start, meeting_node, goal] if meeting_node else [],
            distance=best_distance if best_distance != float('inf') else -1,
            visited_nodes=tracker.visited_count,
            execution_time=execution_time,
            algorithm="Contraction Hierarchies"
        )


# %%
def test_contraction_hierarchies():
    """Тестування Contraction Hierarchies"""
    print("\n⚡ ТЕСТУВАННЯ CONTRACTION HIERARCHIES")
    print("=" * 80)
    
    # Створюємо велику дорожню мережу
    graph, positions = GraphGenerator.create_city_road_network(300, 0.08)
    
    print(f"Граф: {graph.number_of_nodes()} вузлів, {graph.number_of_edges()} ребер")
    
    # Ініціалізація CH
    print("\n⏳ Виконується препроцесинг...")
    ch = ContractionHierarchies(graph)
    preprocessing_time = ch.preprocess()
    
    print(f"Препроцесинг завершено за {preprocessing_time:.2f}ms")
    print(f"Створено {len(ch.shortcuts)} shortcuts")
    
    # Виконуємо кілька запитів
    nodes = list(graph.nodes())
    num_queries = 10
    
    print(f"\n🔍 Виконуємо {num_queries} запитів...")
    
    ch_times = []
    dijkstra_times = []
    ch_visited = []
    dijkstra_visited = []
    
    for _ in range(num_queries):
        start, goal = np.random.choice(nodes, 2, replace=False)
        
        # CH query
        result_ch = ch.query(start, goal)
        ch_times.append(result_ch.execution_time)
        ch_visited.append(result_ch.visited_nodes)
        
        # Dijkstra для порівняння
        result_dijkstra = DijkstraAlgorithm.find_shortest_path(graph, start, goal)
        dijkstra_times.append(result_dijkstra.execution_time)
        dijkstra_visited.append(result_dijkstra.visited_nodes)
    
    print(f"\n📊 Результати ({num_queries} запитів):")
    print(f"CH:")
    print(f"  Середній час запиту: {np.mean(ch_times):.3f}ms")
    print(f"  Середньо відвідано вузлів: {np.mean(ch_visited):.1f}")
    print(f"\nDijkstra:")
    print(f"  Середній час запиту: {np.mean(dijkstra_times):.3f}ms")
    print(f"  Середньо відвідано вузлів: {np.mean(dijkstra_visited):.1f}")
    
    print(f"\n⚡ Прискорення CH:")
    print(f"  Час: {np.mean(dijkstra_times)/np.mean(ch_times):.2f}x швидше")
    print(f"  Вузли: {np.mean(dijkstra_visited)/np.mean(ch_visited):.2f}x менше")
    print(f"\n💾 Компроміс: препроцесинг {preprocessing_time:.2f}ms vs {num_queries} швидких запитів")
    
    return ch, preprocessing_time

ch_results = test_contraction_hierarchies()


# %% [markdown]
#  ## 7. Johnson's Algorithm
# 
# 
# 
#  ### Опис:
# 
#  Алгоритм для знаходження найкоротших шляхів між усіма парами вершин
# 
#  у графах з від'ємними вагами. Комбінує Bellman-Ford та Dijkstra.
# 
# 
# 
#  ### Складність:
# 
#  - **Часова**: O(V² log V + VE)
# 
#  - **Просторова**: O(V²)
# 
# 
# 
#  ### Переваги:
# 
#  - Швидше за Floyd-Warshall для розріджених графів
# 
#  - Працює з від'ємними вагами
# 
# 
# 
#  ### Недоліки:
# 
#  - Складніша реалізація
# 
#  - Все одно повільний для великих графів

# %%
class JohnsonAlgorithm:
    """Реалізація алгоритму Джонсона"""
    
    @staticmethod
    def find_all_shortest_paths(graph: nx.Graph) -> Tuple[Dict, float]:
        """
        Знаходить найкоротші шляхи між усіма парами вершин
        Повертає (словник_відстаней, час_виконання)
        """
        start_time = time.perf_counter()
        
        # Крок 1: Додаємо фіктивну вершину, з'єднану з усіма іншими вагою 0
        augmented_graph = graph.copy()
        dummy_node = max(graph.nodes()) + 1
        for node in graph.nodes():
            augmented_graph.add_edge(dummy_node, node, weight=0)
        
        # Крок 2: Bellman-Ford від фіктивної вершини для знаходження потенціалів
        distances = {node: float('inf') for node in augmented_graph.nodes()}
        distances[dummy_node] = 0
        
        # Релаксація
        for _ in range(len(augmented_graph.nodes()) - 1):
            for u, v in augmented_graph.edges():
                weight = augmented_graph[u][v].get('weight', 1)
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                if distances[v] + weight < distances[u]:
                    distances[u] = distances[v] + weight
        
        # Потенціали h(v)
        h = {node: distances[node] for node in graph.nodes()}
        
        # Крок 3: Перезважування ребер: w'(u,v) = w(u,v) + h(u) - h(v)
        reweighted_graph = graph.copy()
        for u, v in reweighted_graph.edges():
            original_weight = reweighted_graph[u][v].get('weight', 1)
            new_weight = original_weight + h[u] - h[v]
            reweighted_graph[u][v]['weight'] = new_weight
        
        # Крок 4: Dijkstra від кожної вершини на перезваженому графі
        all_distances = {}
        for source in graph.nodes():
            distances_from_source = DijkstraAlgorithm.find_all_shortest_paths(
                reweighted_graph, source
            )
            
            # Крок 5: Відновлення оригінальних відстаней
            all_distances[source] = {}
            for target in graph.nodes():
                if distances_from_source[target] != float('inf'):
                    original_distance = distances_from_source[target] - h[source] + h[target]
                    all_distances[source][target] = original_distance
                else:
                    all_distances[source][target] = float('inf')
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        return all_distances, execution_time


# %%
def test_johnson_vs_floyd_warshall():
    """Порівняння Johnson vs Floyd-Warshall"""
    print("\n🔄 ПОРІВНЯННЯ JOHNSON vs FLOYD-WARSHALL")
    print("=" * 80)
    
    sizes = [20, 50, 100]
    results = []
    
    for size in sizes:
        print(f"\n📊 Розмір графу: {size} вузлів")
        
        # Створюємо розріджений граф
        graph = nx.erdos_renyi_graph(size, 0.1)
        
        # Додаємо випадкові ваги (включаючи від'ємні)
        for u, v in graph.edges():
            graph[u][v]['weight'] = np.random.uniform(-5, 10)
        
        print(f"  Ребер: {graph.number_of_edges()}")
        print(f"  Щільність: {graph.number_of_edges() / (size * (size-1) / 2):.3f}")
        
        # Johnson
        all_dist_johnson, time_johnson = JohnsonAlgorithm.find_all_shortest_paths(graph)
        
        # Floyd-Warshall
        dist_fw, _, time_fw, _ = FloydWarshallAlgorithm.find_all_shortest_paths(graph)# Floyd-Warshall
        
        speedup = time_fw / time_johnson
        
        print(f"  Johnson: {time_johnson:.2f}ms")
        print(f"  Floyd-Warshall: {time_fw:.2f}ms")
        print(f"  Прискорення: {speedup:.2f}x")
        
        results.append({
            'Nodes': size,
            'Edges': graph.number_of_edges(),
            'Density': graph.number_of_edges() / (size * (size-1) / 2),
            'Johnson (ms)': time_johnson,
            'Floyd-Warshall (ms)': time_fw,
            'Speedup': speedup
        })
    
    print("\n" + "="*80)
    print("Висновок: Johnson вигідніший для розріджених графів")
    
    return pd.DataFrame(results)

johnson_comparison = test_johnson_vs_floyd_warshall()


# %% [markdown]
#  ## 8. Візуалізація алгоритмів

# %%
class PathVisualizer:
    """Клас для візуалізації графів та шляхів"""
    
    @staticmethod
    def visualize_path_comparison(graph, positions, paths_dict, title="Порівняння алгоритмів"):
        """
        Візуалізує кілька шляхів на одному графі
        paths_dict: {algorithm_name: (path, color)}
        """
        fig, axes = plt.subplots(1, len(paths_dict), figsize=(7*len(paths_dict), 6))
        
        if len(paths_dict) == 1:
            axes = [axes]
        
        for ax, (algo_name, (path, color)) in zip(axes, paths_dict.items()):
            # Малюємо граф
            nx.draw_networkx_edges(graph, positions, alpha=0.2, ax=ax)
            nx.draw_networkx_nodes(graph, positions, node_size=30, 
                                  node_color='lightgray', ax=ax)
            
            # Виділяємо шлях
            if len(path) > 1:
                path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                nx.draw_networkx_edges(graph, positions, path_edges, 
                                      edge_color=color, width=3, ax=ax)
                
                # Виділяємо вузли шляху
                nx.draw_networkx_nodes(graph, positions, path, 
                                      node_color=color, node_size=100, ax=ax)
                
                # Позначаємо початок і кінець
                nx.draw_networkx_nodes(graph, positions, [path[0]], 
                                      node_color='green', node_size=150, 
                                      node_shape='s', ax=ax)
                nx.draw_networkx_nodes(graph, positions, [path[-1]], 
                                      node_color='red', node_size=150, 
                                      node_shape='s', ax=ax)
            
            ax.set_title(f"{algo_name}\nДовжина шляху: {len(path)} вузлів", 
                        fontsize=12, fontweight='bold')
            ax.axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def visualize_visited_nodes(graph, positions, visited_sets_dict, title="Відвідані вузли"):
        """
        Візуалізує відвідані вузли різними алгоритмами
        visited_sets_dict: {algorithm_name: (visited_set, color)}
        """
        fig, axes = plt.subplots(1, len(visited_sets_dict), 
                                figsize=(7*len(visited_sets_dict), 6))
        
        if len(visited_sets_dict) == 1:
            axes = [axes]
        
        for ax, (algo_name, (visited, color)) in zip(axes, visited_sets_dict.items()):
            # Малюємо граф
            nx.draw_networkx_edges(graph, positions, alpha=0.1, ax=ax)
            
            # Невідвідані вузли
            unvisited = set(graph.nodes()) - visited
            nx.draw_networkx_nodes(graph, positions, unvisited, 
                                  node_size=30, node_color='lightgray', ax=ax)
            
            # Відвідані вузли
            if visited:
                nx.draw_networkx_nodes(graph, positions, list(visited), 
                                      node_size=50, node_color=color, 
                                      alpha=0.6, ax=ax)
            
            ax.set_title(f"{algo_name}\nВідвідано: {len(visited)} вузлів", 
                        fontsize=12, fontweight='bold')
            ax.axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig


# %%
def create_visual_comparison():
    """Створення візуального порівняння алгоритмів"""
    print("\n🎨 СТВОРЕННЯ ВІЗУАЛЬНОГО ПОРІВНЯННЯ")
    print("="*80)
    
    # Створюємо граф підземелля
    graph, positions = GraphGenerator.create_game_dungeon(15, 15, 0.25)
    
    nodes = list(graph.nodes())
    start = nodes[0]
    goal = nodes[-1]
    
    print(f"Сценарій: Ігрове підземелля")
    print(f"Вузлів: {len(nodes)}, Ребер: {graph.number_of_edges()}")
    print(f"Маршрут: {start} → {goal}")
    
    # Запускаємо алгоритми
    result_dijkstra = DijkstraAlgorithm.find_shortest_path(graph, start, goal)
    result_astar = AStarAlgorithm.find_shortest_path(graph, start, goal, positions, 'euclidean')
    result_bidirectional = BidirectionalSearch.find_shortest_path(graph, start, goal)
    
    # Візуалізація 1: Порівняння шляхів
    paths_dict = {
        'Dijkstra': (result_dijkstra.path, 'blue'),
        'A* (Euclidean)': (result_astar.path, 'red'),
        'Bidirectional': (result_bidirectional.path, 'green')
    }
    
    fig1 = PathVisualizer.visualize_path_comparison(
        graph, positions, paths_dict, 
        "Порівняння алгоритмів: Ігрове підземелля"
    )
    plt.savefig('path_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nРезультати:")
    comparison_df = pd.DataFrame([
        {
            'Algorithm': 'Dijkstra',
            'Path Length': len(result_dijkstra.path),
            'Distance': result_dijkstra.distance,
            'Visited': result_dijkstra.visited_nodes,
            'Time (ms)': result_dijkstra.execution_time
        },
        {
            'Algorithm': 'A*',
            'Path Length': len(result_astar.path),
            'Distance': result_astar.distance,
            'Visited': result_astar.visited_nodes,
            'Time (ms)': result_astar.execution_time
        },
        {
            'Algorithm': 'Bidirectional',
            'Path Length': len(result_bidirectional.path),
            'Distance': result_bidirectional.distance,
            'Visited': result_bidirectional.visited_nodes,
            'Time (ms)': result_bidirectional.execution_time
        }
    ])
    
    print(comparison_df.to_string(index=False))

create_visual_comparison()


# %% [markdown]
#  ## 9. Комплексний аналіз продуктивності

# %%
class PerformanceAnalyzer:
    """Клас для комплексного аналізу продуктивності"""
    
    @staticmethod
    def benchmark_algorithms(graph_sizes=[50, 100, 200, 500], connectivity=0.1):
        """
        Бенчмарк всіх алгоритмів на різних розмірах графів
        """
        results = []
        
        for size in graph_sizes:
            print(f"\n📊 Тестування на графі {size} вузлів...")
            
            graph, positions = GraphGenerator.create_city_road_network(size, connectivity)
            nodes = list(graph.nodes())
            start, goal = nodes[0], nodes[-1]
            
            # Dijkstra
            result_dijkstra = DijkstraAlgorithm.find_shortest_path(graph, start, goal)
            
            # A*
            result_astar = AStarAlgorithm.find_shortest_path(
                graph, start, goal, positions, 'euclidean'
            )
            
            # Bidirectional
            result_bidirectional = BidirectionalSearch.find_shortest_path(graph, start, goal)
            
            # Bellman-Ford
            result_bellman, _ = BellmanFordAlgorithm.find_shortest_path(graph, start, goal)
            
            results.append({
                'Size': size,
                'Edges': graph.number_of_edges(),
                'Dijkstra Time': result_dijkstra.execution_time,
                'Dijkstra Visited': result_dijkstra.visited_nodes,
                'A* Time': result_astar.execution_time,
                'A* Visited': result_astar.visited_nodes,
                'Bidirectional Time': result_bidirectional.execution_time,
                'Bidirectional Visited': result_bidirectional.visited_nodes,
                'Bellman-Ford Time': result_bellman.execution_time,
                'Bellman-Ford Visited': result_bellman.visited_nodes
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def plot_performance_comparison(df):
        """Візуалізація результатів бенчмарків"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        algorithms = ['Dijkstra', 'A*', 'Bidirectional', 'Bellman-Ford']
        colors = ['blue', 'red', 'green', 'orange']
        
        # Графік 1: Час виконання
        ax = axes[0, 0]
        for algo, color in zip(algorithms, colors):
            ax.plot(df['Size'], df[f'{algo} Time'], 
                   'o-', label=algo, color=color, linewidth=2, markersize=6)
        ax.set_xlabel('Розмір графу (вузлів)', fontsize=11)
        ax.set_ylabel('Час виконання (мс)', fontsize=11)
        ax.set_title('Порівняння часу виконання', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Графік 2: Відвідані вузли
        ax = axes[0, 1]
        for algo, color in zip(algorithms, colors):
            ax.plot(df['Size'], df[f'{algo} Visited'], 
                   'o-', label=algo, color=color, linewidth=2, markersize=6)
        ax.set_xlabel('Розмір графу (вузлів)', fontsize=11)
        ax.set_ylabel('Відвідані вузли', fontsize=11)
        ax.set_title('Порівняння кількості відвіданих вузлів', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Графік 3: Прискорення відносно Dijkstra (час)
        ax = axes[1, 0]
        for algo, color in zip(['A*', 'Bidirectional'], ['red', 'green']):
            speedup = df['Dijkstra Time'] / df[f'{algo} Time']
            ax.plot(df['Size'], speedup, 
                   'o-', label=f'{algo} vs Dijkstra', color=color, 
                   linewidth=2, markersize=6)
        ax.axhline(y=1, color='blue', linestyle='--', alpha=0.5, label='Dijkstra (baseline)')
        ax.set_xlabel('Розмір графу (вузлів)', fontsize=11)
        ax.set_ylabel('Прискорення (разів)', fontsize=11)
        ax.set_title('Прискорення за часом відносно Dijkstra', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Графік 4: Ефективність (відвідані вузли відносно Dijkstra)
        ax = axes[1, 1]
        for algo, color in zip(['A*', 'Bidirectional'], ['red', 'green']):
            efficiency = df['Dijkstra Visited'] / df[f'{algo} Visited']
            ax.plot(df['Size'], efficiency, 
                   'o-', label=f'{algo} vs Dijkstra', color=color, 
                   linewidth=2, markersize=6)
        ax.axhline(y=1, color='blue', linestyle='--', alpha=0.5, label='Dijkstra (baseline)')
        ax.set_xlabel('Розмір графу (вузлів)', fontsize=11)
        ax.set_ylabel('Ефективність (разів)', fontsize=11)
        ax.set_title('Зменшення відвіданих вузлів відносно Dijkstra', 
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# %%
print("\n🔬 КОМПЛЕКСНИЙ БЕНЧМАРК АЛГОРИТМІВ")
print("="*80)
print("Це може зайняти кілька хвилин...")

benchmark_results = PerformanceAnalyzer.benchmark_algorithms([50, 100, 200, 400])

print("\n📊 Результати бенчмарків:")
print(benchmark_results.to_string(index=False))

fig_performance = PerformanceAnalyzer.plot_performance_comparison(benchmark_results)
plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# %% [markdown]
#  ## 10. Практичні сценарії застосування

# %%
class RealWorldScenarios:
    """Реалістичні сценарії застосування алгоритмів"""
    
    @staticmethod
    def gps_navigation_simulation():
        """Симуляція GPS-навігації в місті"""
        print("\n🗺️ СЦЕНАРІЙ 1: GPS-НАВІГАЦІЯ В МІСТІ")
        print("="*80)
        
        # Створюємо реалістичну дорожню мережу
        graph, positions = GraphGenerator.create_city_road_network(200, 0.12)
        
        print(f"Міська мережа: {graph.number_of_nodes()} перехресть")
        
        # Симулюємо кілька маршрутів
        scenarios = [
            ("Короткий маршрут (сусідні квартали)", 10),
            ("Середній маршрут (через район)", 50),
            ("Довгий маршрут (через місто)", 150)
        ]
        
        results = []
        nodes = list(graph.nodes())
        
        for scenario_name, distance_nodes in scenarios:
            start = nodes[0]
            goal = nodes[min(distance_nodes, len(nodes)-1)]
            
            print(f"\n{scenario_name}")
            print(f"  Від: {start}, До: {goal}")
            
            # A* (основний метод для GPS)
            result_astar = AStarAlgorithm.find_shortest_path(
                graph, start, goal, positions, 'euclidean'
            )
            
            # Dijkstra для порівняння
            result_dijkstra = DijkstraAlgorithm.find_shortest_path(graph, start, goal)
            
            print(f"  A*: {result_astar.distance:.1f} км, "
                  f"{len(result_astar.path)} точок, {result_astar.execution_time:.2f}ms")
            print(f"  Dijkstra: {result_dijkstra.distance:.1f} км, "
                  f"{result_dijkstra.visited_nodes} досліджено")
            print(f"  Економія A*: {result_dijkstra.visited_nodes - result_astar.visited_nodes} "
                  f"вузлів ({(1 - result_astar.visited_nodes/result_dijkstra.visited_nodes)*100:.1f}%)")
            
            results.append({
                'Scenario': scenario_name,
                'Distance': result_astar.distance,
                'A* Time': result_astar.execution_time,
                'A* Visited': result_astar.visited_nodes,
                'Dijkstra Visited': result_dijkstra.visited_nodes,
                'Savings %': (1 - result_astar.visited_nodes/result_dijkstra.visited_nodes)*100
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def social_network_connections():
        """Аналіз зв'язків у соціальній мережі"""
        print("\n👥 СЦЕНАРІЙ 2: СОЦІАЛЬНА МЕРЕЖА")
        print("="*80)
        
        graph = GraphGenerator.create_social_network(500, avg_friends=10)
        
        print(f"Користувачів: {graph.number_of_nodes()}")
        print(f"Зв'язків: {graph.number_of_edges()}")
        
        # Знаходимо ступені розділення між випадковими користувачами
        samples = 20
        separations = []
        
        for _ in range(samples):
            user1, user2 = np.random.choice(list(graph.nodes()), 2, replace=False)
            result = BidirectionalSearch.find_shortest_path(graph, user1, user2)
            if len(result.path) > 0:
                separations.append(len(result.path) - 1)
        
        print(f"\nСтупені розділення (вибірка {samples} пар):")
        print(f"  Середній: {np.mean(separations):.2f}")
        print(f"  Медіана: {np.median(separations):.0f}")
        print(f"  Мін/Макс: {min(separations)}/{max(separations)}")
        
        # Розподіл
        unique, counts = np.unique(separations, return_counts=True)
        print("\nРозподіл:")
        for degree, count in zip(unique, counts):
            print(f"  {degree} ступенів: {count} пар ({count/samples*100:.0f}%)")
        
        return separations
    
    @staticmethod
    def supply_chain_optimization():
        """Оптимізація ланцюга постачання"""
        print("\n📦 СЦЕНАРІЙ 3: ЛОГІСТИКА ТА ЛАНЦЮГ ПОСТАЧАННЯ")
        print("="*80)
        
        graph = GraphGenerator.create_logistics_network(15, 30)
        
        warehouses = [n for n in graph.nodes() if graph.nodes[n].get('node_type') == 'warehouse']
        stores = [n for n in graph.nodes() if graph.nodes[n].get('node_type') == 'store']
        
        print(f"Складів: {len(warehouses)}")
        print(f"Магазинів: {len(stores)}")
        
        # Для кожного магазину знаходимо найближчий склад
        print("\nОптимізація маршрутів доставки...")
        
        total_distance = 0
        deliveries = []
        
        for store in stores[:10]:  # Беремо перші 10 для прикладу
            best_warehouse = None
            best_distance = float('inf')
            
            for warehouse in warehouses:
                result = DijkstraAlgorithm.find_shortest_path(graph, warehouse, store)
                if result.distance < best_distance:
                    best_distance = result.distance
                    best_warehouse = warehouse
            
            total_distance += best_distance
            deliveries.append({
                'Store': store,
                'Warehouse': best_warehouse,
                'Distance': best_distance
            })
        
        print(f"Загальна відстань для 10 доставок: {total_distance:.1f} км")
        print(f"Середня відстань: {total_distance/10:.1f} км")
        
        return pd.DataFrame(deliveries)
    
    @staticmethod
    def game_pathfinding():
        """Пошук шляху для ігрових персонажів"""
        print("\n🎮 СЦЕНАРІЙ 4: ІГРОВА ІНДУСТРІЯ - ПОШУК ШЛЯХУ NPC")
        print("="*80)
        
        # Велике підземелля
        graph, positions = GraphGenerator.create_game_dungeon(30, 30, 0.3)
        
        print(f"Розмір підземелля: 30x30")
        print(f"Прохідних клітинок: {graph.number_of_nodes()}")
        
        # Тестуємо різні відстані
        nodes = list(graph.nodes())
        distances = [10, 50, 100, 200]
        
        results = []
        
        for dist in distances:
            if dist < len(nodes):
                start = nodes[0]
                goal = nodes[min(dist, len(nodes)-1)]
                
                # A* - стандарт для ігор
                result_astar = AStarAlgorithm.find_shortest_path(
                    graph, start, goal, positions, 'manhattan'
                )
                
                # Dijkstra для порівняння
                result_dijkstra = DijkstraAlgorithm.find_shortest_path(graph, start, goal)
                
                results.append({
                    'Distance': dist,
                    'A* Time (ms)': result_astar.execution_time,
                    'A* Visited': result_astar.visited_nodes,
                    'Dijkstra Visited': result_dijkstra.visited_nodes,
                    'Efficiency': result_dijkstra.visited_nodes / result_astar.visited_nodes
                })
                
                print(f"\nВідстань ~{dist} клітинок:")
                print(f"  A*: {result_astar.execution_time:.3f}ms, "
                      f"{result_astar.visited_nodes} вузлів")
                print(f"  Ефективність: {result_dijkstra.visited_nodes / result_astar.visited_nodes:.1f}x")
        
        print(f"\n💡 Висновок: A* з Manhattan heuristic оптимальний для сіток")
        
        return pd.DataFrame(results)


# %%
# Виконання всіх практичних сценаріїв

gps_results = RealWorldScenarios.gps_navigation_simulation()
social_results = RealWorldScenarios.social_network_connections()
logistics_results = RealWorldScenarios.supply_chain_optimization()
game_results = RealWorldScenarios.game_pathfinding()


# %% [markdown]
#  ## 11. Підсумкова таблиця та рекомендації

# %%
def create_algorithm_summary():
    """Створює підсумкову таблицю з рекомендаціями"""
    print("\n📋 ПІДСУМКОВА ТАБЛИЦЯ АЛГОРИТМІВ")
    print("="*80)
    
    summary_data = [
        {
            'Алгоритм': 'Dijkstra',
            'Складність': 'O(E log V)',
            'Від\'ємні ваги': 'Ні',
            'Використання': 'GPS, маршрутизація, базовий пошук',
            'Переваги': 'Оптимальний, надійний',
            'Недоліки': 'Досліджує багато вузлів',
            'Коли використовувати': 'Коли потрібна гарантія оптимальності'
        },
        {
            'Алгоритм': 'A*',
            'Складність': 'O(E) середнє',
            'Від\'ємні ваги': 'Ні',
            'Використання': 'Ігри, робототехніка, GPS з координатами',
            'Переваги': 'Дуже швидкий з хорошою евристикою',
            'Недоліки': 'Потребує евристичну функцію',
            'Коли використовувати': 'Коли є координати або евристика'
        },
        {
            'Алгоритм': 'Bellman-Ford',
            'Складність': 'O(V·E)',
            'Від\'ємні ваги': 'Так',
            'Використання': 'Арбітраж, мережі з від\'ємними вагами',
            'Переваги': 'Працює з від\'ємними вагами',
            'Недоліки': 'Повільний',
            'Коли використовувати': 'Коли є від\'ємні ваги'
        },
        {
            'Алгоритм': 'Floyd-Warshall',
            'Складність': 'O(V³)',
            'Від\'ємні ваги': 'Так',
            'Використання': 'Малі графи, всі пари шляхів',
            'Переваги': 'Знаходить всі шляхи одразу',
            'Недоліки': 'Дуже повільний для великих графів',
            'Коли використовувати': 'Графи <500 вузлів, потрібні всі шляхи'
        },
        {
            'Алгоритм': 'Bidirectional',
            'Складність': 'O(b^(d/2))',
            'Від\'ємні ваги': 'Ні',
            'Використання': 'Великі графи, відомий старт і фініш',
            'Переваги': 'Експоненційне прискорення',
            'Недоліки': 'Складна реалізація',
            'Коли використовувати': 'Великі графи з відомою метою'
        },
        {
            'Алгоритм': 'Contraction H.',
            'Складність': 'Query: O(log n)',
            'Від\'ємні ваги': 'Ні',
            'Використання': 'Навігаційні системи (OSRM)',
            'Переваги': 'Надшвидкі запити',
            'Недоліки': 'Довгий препроцесинг',
            'Коли використовувати': 'Багато запитів на статичному графі'
        },
        {
            'Алгоритм': 'Johnson',
            'Складність': 'O(V²log V + VE)',
            'Від\'ємні ваги': 'Так',
            'Використання': 'Всі пари, розріджені графи',
            'Переваги': 'Швидше Floyd-Warshall для розріджених',
            'Недоліки': 'Складна реалізація',
            'Коли використовувати': 'Всі пари + розріджений граф'
        }
    ]
    
    df = pd.DataFrame(summary_data)
    
    # Виводимо таблицю частинами для кращої читабельності
    print("\nОсновні характеристики:")
    print(df[['Алгоритм', 'Складність', 'Від\'ємні ваги']].to_string(index=False))
    
    print("\n\nПрактичне застосування:")
    print(df[['Алгоритм', 'Використання', 'Коли використовувати']].to_string(index=False))
    
    return df

algorithm_summary = create_algorithm_summary()


# %%
def create_decision_tree():
    """Створює дерево рішень для вибору алгоритму"""
    print("\n🌳 ДЕРЕВО РІШЕНЬ: ЯКИЙ АЛГОРИТМ ВИБРАТИ?")
    print("="*80)
    
    print("""
    ┌─ Чи є від'ємні ваги?
    │
    ├─ ТАК
    │  │
    │  ├─ Один шлях → Bellman-Ford
    │  │
    │  └─ Всі пари шляхів
    │     │
    │     ├─ Маленький граф (<100 вузлів) → Floyd-Warshall
    │     │
    │     └─ Великий граф → Johnson
    │
    └─ НІ (тільки позитивні ваги)
       │
       ├─ Чи потрібні всі пари шляхів?
       │  │
       │  ├─ ТАК
       │  │  │
       │  │  ├─ Маленький граф (<200 вузлів) → Floyd-Warshall
       │  │  │
       │  │  └─ Великий граф → Johnson або багато разів Dijkstra
       │  │
       │  └─ НІ (один шлях)
       │     │
       │     ├─ Чи є географічні координати/евристика?
       │     │  │
       │     │  ├─ ТАК → A* (найшвидший!)
       │     │  │
       │     │  └─ НІ → Діаметр графу великий?
       │     │     │
       │     │     ├─ ТАК → Bidirectional Search
       │     │     │
       │     │     └─ НІ → Dijkstra
       │
       └─ Багато запитів на статичному графі?
          │
          └─ ТАК → Contraction Hierarchies (препроцесинг + швидкі запити)
    
    """)
    
    print("\n💡 ШВИДКІ РЕКОМЕНДАЦІЇ ЗА ДОМЕНАМИ:")
    print("-" * 80)
    
    domain_recommendations = {
        'GPS/Навігація': 'A* з Euclidean heuristic або Contraction Hierarchies',
        'Соціальні мережі': 'Bidirectional Search (відомі start/goal)',
        'Ігри (сітки)': 'A* з Manhattan heuristic',
        'Логістика': 'Dijkstra або Floyd-Warshall (малі мережі)',
        'Фінанси/Арбітраж': 'Bellman-Ford (від\'ємні ваги)',
        'Мережева маршрутизація': 'Dijkstra або Contraction Hierarchies',
        'Аналіз графів': 'Floyd-Warshall або Johnson (всі пари)'
    }
    
    for domain, recommendation in domain_recommendations.items():
        print(f"  {domain:.<30} {recommendation}")

create_decision_tree()


# %% [markdown]
#  ## 12. Візуалізація масштабованості

# %%
def create_scalability_analysis():
    """Аналіз масштабованості алгоритмів"""
    print("\n📈 АНАЛІЗ МАСШТАБОВАНОСТІ")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Теоретичні складності
    n = np.logspace(1, 3, 50)  # від 10 до 1000
    
    complexities = {
        'A* O(E)': n * np.log2(n) * 0.5,  # приблизно
        'Dijkstra O(E log V)': n * np.log2(n),
        'Bellman-Ford O(VE)': n**2,
        'Floyd-Warshall O(V³)': n**3,
    }
    
    # Графік 1: Абсолютна складність
    ax = axes[0, 0]
    for name, values in complexities.items():
        ax.plot(n, values, label=name, linewidth=2)
    ax.set_xlabel('Кількість вузлів')
    ax.set_ylabel('Операції')
    ax.set_title('Теоретична складність алгоритмів')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Графік 2: Відносне прискорення A* vs Dijkstra
    ax = axes[0, 1]
    speedup_theoretical = complexities['Dijkstra O(E log V)'] / complexities['A* O(E)']
    ax.plot(n, speedup_theoretical, 'r-', linewidth=2, label='Теоретичне')
    ax.axhline(y=2, color='green', linestyle='--', alpha=0.5, label='2x прискорення')
    ax.set_xlabel('Кількість вузлів')
    ax.set_ylabel('Прискорення (разів)')
    ax.set_title('Прискорення A* відносно Dijkstra')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Графік 3: Час виконання для різних типів графів
    ax = axes[1, 0]
    
    graph_types = ['Розріджений\n(E~V)', 'Середній\n(E~V log V)', 'Щільний\n(E~V²)']
    dijkstra_times = [100, 200, 500]
    astar_times = [50, 80, 200]
    bidirectional_times = [40, 90, 250]
    
    x = np.arange(len(graph_types))
    width = 0.25
    
    ax.bar(x - width, dijkstra_times, width, label='Dijkstra', color='blue', alpha=0.7)
    ax.bar(x, astar_times, width, label='A*', color='red', alpha=0.7)
    ax.bar(x + width, bidirectional_times, width, label='Bidirectional', color='green', alpha=0.7)
    
    ax.set_xlabel('Тип графу')
    ax.set_ylabel('Відносний час')
    ax.set_title('Продуктивність на різних типах графів')
    ax.set_xticks(x)
    ax.set_xticklabels(graph_types)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Графік 4: Пам'ять vs швидкість компроміс
    ax = axes[1, 1]
    
    algorithms_memory = {
        'A*': (1.5, 0.3, 'red'),  # (швидкість, пам'ять)
        'Dijkstra': (1.0, 0.5, 'blue'),
        'Bidirectional': (1.8, 0.6, 'green'),
        'Bellman-Ford': (0.2, 0.5, 'orange'),
        'Floyd-Warshall': (0.1, 2.0, 'purple'),
        'Contraction H.': (3.0, 1.5, 'brown')
    }
    
    for name, (speed, memory, color) in algorithms_memory.items():
        ax.scatter(memory, speed, s=300, c=color, alpha=0.6, edgecolors='black', linewidth=2)
        ax.annotate(name, (memory, speed), fontsize=10, ha='center', va='center')
    
    ax.set_xlabel('Використання пам\'яті (відносно)', fontsize=11)
    ax.set_ylabel('Швидкість (відносно)', fontsize=11)
    ax.set_title('Компроміс пам\'ять vs швидкість', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scalability_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nВисновки:")
    print("  • A* найефективніший при наявності евристики")
    print("  • Bidirectional чудовий для великих графів")
    print("  • Floyd-Warshall працює тільки для малих графів")
    print("  • Contraction Hierarchies - компроміс: багато пам'яті, але надшвидко")

create_scalability_analysis()


# %% [markdown]
#  ## 13. Експертні поради та підводні камені

# %%
def print_expert_tips():
    """Експертні поради та типові помилки"""
    print("\n🎓 ЕКСПЕРТНІ ПОРАДИ ТА ПІДВОДНІ КАМЕНІ")
    print("="*80)
    
    print("\n1️⃣ ВИБІР АЛГОРИТМУ")
    print("-" * 40)
    print("""
    ✓ Завжди починайте з аналізу даних:
      - Розмір графу (V, E)
      - Тип графу (розріджений/щільний)
      - Наявність координат
      - Чи є від'ємні ваги
      - Кількість запитів
    
    ✗ Типова помилка: використання Floyd-Warshall для великих графів
      → Використовуйте Johnson або багато разів Dijkstra
    """)
    
    print("\n2️⃣ ОПТИМІЗАЦІЯ A*")
    print("-" * 40)
    print("""
    ✓ Вибір правильної евристики критичний:
      - Euclidean: географічні координати
      - Manhattan: сітки (ігри, лабіринти)
      - Chebyshev: діагональний рух
      - Октильна: діагональ + прямо
    
    ✓ Евристика має бути допустимою (admissible):
      h(n) ≤ реальна відстань до мети
    
    ✗ Типова помилка: занадто оптимістична евристика
      → Порушує оптимальність результату
    """)
    
    print("\n3️⃣ ПРОДУКТИВНІСТЬ")
    print("-" * 40)
    print("""
    ✓ Використовуйте правильні структури даних:
      - Бінарна купа: для Dijkstra
      - Fibonacci heap: теоретично краще, практично рідко
      - Hash map: для visited/distances
    
    ✓ Ранній вихід (early exit):
      - Зупиняйте Dijkstra при досягненні мети
      - Не обробляйте всі вузли якщо не потрібно
    
    ✗ Типова помилка: використання списку замість купи
      → Складність O(V²) замість O(E log V)
    """)
    
    print("\n4️⃣ ПАРАЛЕЛІЗАЦІЯ")
    print("-" * 40)
    print("""
    ✓ Що можна паралелізувати:
      - Bidirectional search (2 паралельні пошуки)
      - Багато незалежних запитів
      - Contraction Hierarchies (препроцесинг)
    
    ✗ Складно паралелізувати:
      - Один запит Dijkstra/A*
      - Bellman-Ford (послідовні релаксації)
    """)
    
    print("\n5️⃣ СПЕЦІАЛЬНІ ВИПАДКИ")
    print("-" * 40)
    print("""
    ✓ Невеликі графи (<100 вузлів):
      - Простий Dijkstra часто найкращий
      - Overhead складних методів не виправданий
    
    ✓ Дуже великі графи (>1M вузлів):
      - Contraction Hierarchies обов'язковий
      - Або ієрархічні методи (ALT)
    
    ✓ Динамічні графи (змінюються ваги):
      - Уникайте препроцесингу
      - Використовуйте A* або Dijkstra
      - Інкрементальні алгоритми (D* Lite)
    """)
    
    print("\n6️⃣ НАЛАГОДЖЕННЯ")
    print("-" * 40)
    print("""
    ✓ Завжди перевіряйте:
      - Чи шлях існує (граф зв'язний?)
      - Чи всі ваги позитивні (якщо використовуєте Dijkstra)
      - Чи евристика допустима (для A*)
      - Чи немає від'ємних циклів
    
    ✓ Візуалізуйте:
      - Відвідані вузли
      - Знайдений шлях
      - Порядок обробки
    """)
    
    print("\n7️⃣ ПРАКТИЧНІ ОПТИМІЗАЦІЇ")
    print("-" * 40)
    print("""
    ✓ Bidirectional A*:
      - Комбінує переваги обох підходів
      - Складніша реалізація, але швидше
    
    ✓ Jump Point Search (для сіток):
      - Оптимізація A* для uniform-cost сіток
      - До 10x швидше на великих сітках
    
    ✓ Landmark-based methods (ALT):
      - Використання "орієнтирів" для евристики
      - Хороша альтернатива CH
    """)

print_expert_tips()


# %% [markdown]
#  ## 14. Фінальне комплексне порівняння

# %%
def create_final_comparison():
    """Фінальне всеосяжне порівняння"""
    print("\n🏆 ФІНАЛЬНЕ КОМПЛЕКСНЕ ПОРІВНЯННЯ")
    print("="*80)
    
    # Створюємо різні типи графів
    scenarios = {
        'Маленький щільний': (50, 0.3),
        'Середній розріджений': (200, 0.08),
        'Великий розріджений': (500, 0.05)
    }
    
    all_results = []
    
    for scenario_name, (size, connectivity) in scenarios.items():
        print(f"\n{'='*80}")
        print(f"Сценарій: {scenario_name} ({size} вузлів, connectivity={connectivity})")
        print('='*80)
        
        graph, positions = GraphGenerator.create_city_road_network(size, connectivity)
        nodes = list(graph.nodes())
        start, goal = nodes[0], nodes[-1]
        
        print(f"Граф: {graph.number_of_nodes()} вузлів, {graph.number_of_edges()} ребер")
        print(f"Щільність: {2*graph.number_of_edges()/(size*(size-1)):.4f}")
        
        # Тестуємо всі алгоритми
        algorithms = []
        
        # Dijkstra
        result = DijkstraAlgorithm.find_shortest_path(graph, start, goal)
        algorithms.append(('Dijkstra', result))
        
        # A*
        result = AStarAlgorithm.find_shortest_path(graph, start, goal, positions, 'euclidean')
        algorithms.append(('A* (Euclidean)', result))
        
        # Bidirectional
        result = BidirectionalSearch.find_shortest_path(graph, start, goal)
        algorithms.append(('Bidirectional', result))
        
        # Bellman-Ford (тільки для малих графів)
        if size <= 200:
            result, _ = BellmanFordAlgorithm.find_shortest_path(graph, start, goal)
            algorithms.append(('Bellman-Ford', result))
        
        # Виводимо результати
        print(f"\n{'Алгоритм':<20} {'Час (ms)':<12} {'Відвідано':<12} {'Відстань':<12} {'Ефективність'}")
        print('-'*80)
        
        dijkstra_visited = algorithms[0][1].visited_nodes
        
        for name, result in algorithms:
            efficiency = dijkstra_visited / result.visited_nodes if result.visited_nodes > 0 else 1
            print(f"{name:<20} {result.execution_time:>10.3f}  {result.visited_nodes:>10}  "
                  f"{result.distance:>10.2f}  {efficiency:>6.2f}x")
            
            all_results.append({
                'Scenario': scenario_name,
                'Size': size,
                'Algorithm': name,
                'Time (ms)': result.execution_time,
                'Visited': result.visited_nodes,
                'Distance': result.distance,
                'Efficiency': efficiency
            })
    
    return pd.DataFrame(all_results)

final_comparison = create_final_comparison()


# %%
# Візуалізація фінального порівняння
def visualize_final_comparison(df):
    """Візуалізація фінальних результатів"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    scenarios = df['Scenario'].unique()
    
    # Графік 1: Час виконання
    ax = axes[0, 0]
    for scenario in scenarios:
        scenario_data = df[df['Scenario'] == scenario]
        ax.bar(scenario_data['Algorithm'], scenario_data['Time (ms)'], 
               label=scenario, alpha=0.7)
    ax.set_ylabel('Час виконання (мс)')
    ax.set_title('Порівняння часу виконання')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)
    
    # Графік 2: Відвідані вузли
    ax = axes[0, 1]
    for scenario in scenarios:
        scenario_data = df[df['Scenario'] == scenario]
        ax.bar(scenario_data['Algorithm'], scenario_data['Visited'], 
               label=scenario, alpha=0.7)
    ax.set_ylabel('Відвідані вузли')
    ax.set_title('Порівняння кількості відвіданих вузлів')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    ax.tick_params(axis='x', rotation=45)
    
    # Графік 3: Ефективність
    ax = axes[1, 0]
    algorithms = ['A* (Euclidean)', 'Bidirectional']
    for algo in algorithms:
        algo_data = df[df['Algorithm'] == algo].groupby('Scenario')['Efficiency'].mean()
        ax.plot(algo_data.index, algo_data.values, 'o-', label=algo, linewidth=2, markersize=8)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Dijkstra (baseline)')
    ax.set_ylabel('Ефективність (разів)')
    ax.set_title('Ефективність відносно Dijkstra')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Графік 4: Час vs Ефективність
    ax = axes[1, 1]
    for scenario in scenarios:
        scenario_data = df[(df['Scenario'] == scenario) & (df['Algorithm'] != 'Dijkstra')]
        ax.scatter(scenario_data['Time (ms)'], scenario_data['Efficiency'], 
                  s=200, alpha=0.6, label=scenario)
        
        for _, row in scenario_data.iterrows():
            ax.annotate(row['Algorithm'].split()[0], 
                       (row['Time (ms)'], row['Efficiency']),
                       fontsize=8, ha='center')
    
    ax.set_xlabel('Час виконання (мс)')
    ax.set_ylabel('Ефективність (разів)')
    ax.set_title('Компроміс час vs ефективність')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('final_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

visualize_final_comparison(final_comparison)


# %% [markdown]
#  ## 15. Висновки та практичні рекомендації

# %%
print("\n" + "="*80)
print("🎯 ГОЛОВНІ ВИСНОВКИ ТА РЕКОМЕНДАЦІЇ")
print("="*80)

print("""
📊 РЕЗУЛЬТАТИ ДОСЛІДЖЕННЯ:

1. A* - БЕЗУМОВНИЙ ЛІДЕР для просторового пошуку
   ✓ 2-5x менше відвіданих вузлів ніж Dijkstra
   ✓ Оптимальний результат при допустимій евристиці
   ✓ Застосування: GPS, ігри, робототехніка

2. Bidirectional Search - ЧУДОВИЙ ВИБІР для великих графів
   ✓ Експоненційне зменшення простору пошуку
   ✓ Працює без евристики
   ✓ Застосування: соціальні мережі, великі мережі

3. Dijkstra - НАДІЙНИЙ БАЗОВИЙ ВИБІР
   ✓ Гарантує оптимальність
   ✓ Простий у реалізації
   ✓ Застосування: коли немає кращої альтернативи

4. Contraction Hierarchies - ДЛЯ ПРОДАКШН СИСТЕМ
   ✓ Мікросекундні запити після препроцесингу
   ✓ Основа OSRM, Google Maps
   ✓ Застосування: навігаційні системи

5. Bellman-Ford - СПЕЦІАЛІЗОВАНИЙ ІНСТРУМЕНТ
   ✓ Єдиний варіант для від'ємних ваг
   ✓ Виявляє від'ємні цикли
   ✓ Застосування: арбітраж, фінансові графи

6. Floyd-Warshall/Johnson - ДЛЯ АНАЛІЗУ ГРАФІВ
   ✓ Всі пари шляхів одночасно
   ✓ Тільки для малих/середніх графів
   ✓ Застосування: аналіз мереж, метрики графів

""")

print("\n💡 ПРАКТИЧНІ РЕКОМЕНДАЦІЇ ЗА РОЗМІРОМ:")
print("-"*80)
print("""
Малі графи (<100 вузлів):
  → Dijkstra або простий BFS (overhead складних методів не виправданий)

Середні графи (100-1000 вузлів):
  → A* якщо є координати
  → Bidirectional для загального випадку
  → Floyd-Warshall для всіх пар

Великі графи (>1000 вузлів):
  → A* з добрим heuristic
  → Bidirectional обов'язково
  → Contraction Hierarchies для багатьох запитів

Величезні графи (>100k вузлів):
  → Тільки Contraction Hierarchies або ALT
  → Розподілені алгоритми
  → Approximate shortest paths
""")

print("\n🎮 РЕКОМЕНДАЦІЇ ЗА ЗАСТОСУВАННЯМ:")
print("-"*80)
print("""
Реал-тайм системи (ігри, робототехніка):
  → A* з Manhattan/Euclidean
  → Jump Point Search для сіток
  → Ліміт часу: <1ms на запит

Навігаційні системи:
  → Contraction Hierarchies (OSRM)
  → ALT для динамічних ситуацій
  → A* для локальних маршрутів

Аналіз графів:
  → Floyd-Warshall (<500 вузлів)
  → Johnson (>500 вузлів, розріджений)
  → Багато разів Dijkstra (щільний)

Соціальні мережі:
  → Bidirectional Search
  → BFS для невагових графів
  → Приближені алгоритми для scale
""")

print("\n⚠️ ТИПОВІ ПОМИЛКИ ТА ЯК ЇХ УНИКНУТИ:")
print("-"*80)
print("""
1. Використання Floyd-Warshall на великих графах
   → Перейдіть на Johnson або багаторазовий Dijkstra

2. Dijkstra з від'ємними вагами
   → Використовуйте Bellman-Ford

3. A* без перевірки допустимості евристики
   → Завжди перевіряйте: h(n) ≤ реальна відстань

4. Ігнорування early exit в Dijkstra
   → Зупиняйтесь при досягненні мети

5. Неправильна структура даних
   → Використовуйте heap, не список

6. Відсутність кешування для багатьох запитів
   → Розгляньте Contraction Hierarchies
""")

print("\n🚀 МАЙБУТНІ НАПРЯМКИ:")
print("-"*80)
print("""
- Machine Learning для пошуку шляхів (Neural A*)
- Квантові алгоритми для графів
- GPU-прискорені версії для масивних графів
- Динамічні алгоритми для графів що змінюються
- Гібридні підходи (ML + класичні алгоритми)
""")

print("\n✅ ГОТОВО!")
print("="*80)
print("""
Цей notebook містить:
  • 7 алгоритмів пошуку шляхів
  • 6+ практичних сценаріїв
  • Детальне порівняння продуктивності
  • Візуалізації та графіки
  • Експертні поради
  • Дерево рішень для вибору алгоритму

Всі результати збережені як PNG високої якості.
Матеріал готовий для навчального курсу або презентації!
""")


# %%
# %%


