"""
Модуль жадібних алгоритмів на графах (Дейкстра, Прім, Крускал)
Author: Educational Tutorial
Python version: 3.8+
"""

import heapq
from typing import Dict, List, Tuple, Set, Optional
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Edge:
    """Представлення ребра графа"""
    start: str
    end: str
    weight: float
    
    def __lt__(self, other):
        return self.weight < other.weight


class UnionFind:
    """Структура даних Union-Find для алгоритму Крускала"""
    
    def __init__(self, vertices: List[str]):
        self.parent = {v: v for v in vertices}
        self.rank = {v: 0 for v in vertices}
    
    def find(self, vertex: str) -> str:
        """Знаходить корінь компоненти з стисканням шляху"""
        if self.parent[vertex] != vertex:
            self.parent[vertex] = self.find(self.parent[vertex])
        return self.parent[vertex]
    
    def union(self, v1: str, v2: str) -> bool:
        """Об'єднує дві компоненти. Повертає True, якщо об'єднання відбулося"""
        root1, root2 = self.find(v1), self.find(v2)
        
        if root1 == root2:
            return False  # Вже в одній компоненті
        
        # Union by rank
        if self.rank[root1] < self.rank[root2]:
            self.parent[root1] = root2
        elif self.rank[root1] > self.rank[root2]:
            self.parent[root2] = root1
        else:
            self.parent[root2] = root1
            self.rank[root1] += 1
        
        return True


class GraphAlgorithms:
    """Клас для демонстрації жадібних алгоритмів на графах"""
    
    @staticmethod
    def dijkstra(graph: Dict[str, List[Tuple[str, float]]], start: str) -> Tuple[Dict[str, float], Dict[str, Optional[str]]]:
        """
        Алгоритм Дейкстри для пошуку найкоротших шляхів
        
        Args:
            graph: Граф у вигляді словника списків суміжності
            start: Початкова вершина
            
        Returns:
            Tuple[Dict[str, float], Dict[str, Optional[str]]]: Відстані та попередники
        """
        distances = {vertex: float('inf') for vertex in graph}
        distances[start] = 0
        previous = {vertex: None for vertex in graph}
        visited = set()
        
        # Пріоритетна черга: (відстань, вершина)
        pq = [(0, start)]
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            # Жадібно оновлюємо відстані до сусідів
            for neighbor, weight in graph.get(current, []):
                if neighbor not in visited:
                    new_dist = current_dist + weight
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        previous[neighbor] = current
                        heapq.heappush(pq, (new_dist, neighbor))
        
        return distances, previous
    
    @staticmethod
    def prim_mst(graph: Dict[str, List[Tuple[str, float]]]) -> Tuple[List[Edge], float]:
        """
        Алгоритм Пріма для пошуку мінімального остовного дерева
        
        Args:
            graph: Граф у вигляді словника списків суміжності
            
        Returns:
            Tuple[List[Edge], float]: Список ребер MST та їх загальна вага
        """
        if not graph:
            return [], 0.0
        
        # Починаємо з першої вершини
        start = next(iter(graph))
        mst_edges = []
        visited = {start}
        total_weight = 0.0
        
        # Пріоритетна черга ребер: (вага, початок, кінець)
        pq = []
        for neighbor, weight in graph.get(start, []):
            heapq.heappush(pq, (weight, start, neighbor))
        
        while pq and len(visited) < len(graph):
            weight, start_v, end_v = heapq.heappop(pq)
            
            if end_v in visited:
                continue
            
            # Жадібно додаємо найменше ребро до MST
            mst_edges.append(Edge(start_v, end_v, weight))
            visited.add(end_v)
            total_weight += weight
            
            # Додаємо нові ребра до черги
            for neighbor, edge_weight in graph.get(end_v, []):
                if neighbor not in visited:
                    heapq.heappush(pq, (edge_weight, end_v, neighbor))
        
        return mst_edges, total_weight
    
    @staticmethod
    def kruskal_mst(vertices: List[str], edges: List[Edge]) -> Tuple[List[Edge], float]:
        """
        Алгоритм Крускала для пошуку мінімального остовного дерева
        
        Args:
            vertices: Список вершин графа
            edges: Список усіх ребер графа
            
        Returns:
            Tuple[List[Edge], float]: Список ребер MST та їх загальна вага
        """
        # Сортуємо ребра за вагою (жадібна стратегія)
        sorted_edges = sorted(edges)
        
        mst_edges = []
        total_weight = 0.0
        uf = UnionFind(vertices)
        
        for edge in sorted_edges:
            # Жадібно додаємо найменше ребро, якщо воно не створює цикл
            if uf.union(edge.start, edge.end):
                mst_edges.append(edge)
                total_weight += edge.weight
                
                # MST для n вершин має n-1 ребер
                if len(mst_edges) == len(vertices) - 1:
                    break
        
        return mst_edges, total_weight


class TSPSolver:
    """Клас для демонстрації жадібних евристик для задачі комівояжера"""
    
    @staticmethod
    def nearest_neighbor_tsp(distances: Dict[Tuple[str, str], float], start: str) -> Tuple[List[str], float]:
        """
        Жадібна евристика "найближчий сусід" для TSP
        
        Args:
            distances: Словник відстаней між містами
            start: Початкове місто
            
        Returns:
            Tuple[List[str], float]: Маршрут та його довжина
        """
        # Знаходимо всі міста
        cities = set()
        for city1, city2 in distances.keys():
            cities.add(city1)
            cities.add(city2)
        
        if start not in cities:
            raise ValueError(f"Початкове місто {start} не знайдено в графі")
        
        route = [start]
        unvisited = cities - {start}
        total_distance = 0.0
        current_city = start
        
        # Жадібно обираємо найближче невідвідане місто
        while unvisited:
            nearest_city = None
            min_distance = float('inf')
            
            for city in unvisited:
                # Відстань може бути задана в обох напрямках
                dist = distances.get((current_city, city)) or distances.get((city, current_city))
                if dist is not None and dist < min_distance:
                    min_distance = dist
                    nearest_city = city
            
            if nearest_city is None:
                break
            
            route.append(nearest_city)
            unvisited.remove(nearest_city)
            total_distance += min_distance
            current_city = nearest_city
        
        # Повертаємося до початкової точки
        return_dist = distances.get((current_city, start)) or distances.get((start, current_city))
        if return_dist is not None:
            route.append(start)
            total_distance += return_dist
        
        return route, total_distance


class GraphVisualizer:
    """Клас для візуалізації графових алгоритмів"""
    
    @staticmethod
    def visualize_dijkstra(graph: Dict[str, List[Tuple[str, float]]], start: str):
        """Візуалізує результат алгоритму Дейкстри"""
        distances, previous = GraphAlgorithms.dijkstra(graph, start)
        
        # Створюємо NetworkX граф
        G = nx.Graph()
        for vertex, neighbors in graph.items():
            for neighbor, weight in neighbors:
                G.add_edge(vertex, neighbor, weight=weight)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Лівий графік: оригінальний граф
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, ax=ax1, with_labels=True, node_color='lightblue', 
                node_size=1000, font_size=12, font_weight='bold')
        
        # Додаємо ваги ребер
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax1)
        ax1.set_title(f'Оригінальний граф\nПочаткова вершина: {start}')
        
        # Правий графік: дерево найкоротших шляхів
        shortest_path_edges = []
        for vertex, prev in previous.items():
            if prev is not None:
                shortest_path_edges.append((prev, vertex))
        
        # Малюємо граф з виділеними найкоротшими шляхами
        nx.draw(G, pos, ax=ax2, with_labels=True, node_color='lightgreen',
                node_size=1000, font_size=12, font_weight='bold', edge_color='lightgray')
        
        # Виділяємо найкоротші шляхи
        if shortest_path_edges:
            nx.draw_networkx_edges(G, pos, shortest_path_edges, ax=ax2,
                                 edge_color='red', width=3)
        
        # Додаємо відстані як підписи до вершин
        for vertex, (x, y) in pos.items():
            dist = distances[vertex]
            dist_text = f"{dist:.1f}" if dist != float('inf') else "∞"
            ax2.text(x, y-0.15, f"d={dist_text}", ha='center', va='top',
                    bbox={'boxstyle': 'round,pad=0.3', 'facecolor': 'yellow', 'alpha': 0.7})
        
        ax2.set_title('Дерево найкоротших шляхів\n(червоні ребра)')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def visualize_mst_comparison(graph: Dict[str, List[Tuple[str, float]]]):
        """Порівнює алгоритми Пріма та Крускала для MST"""
        # Підготовка даних для Крускала
        vertices = list(graph.keys())
        edges = []
        for vertex, neighbors in graph.items():
            for neighbor, weight in neighbors:
                # Уникаємо дублювання ребер
                if vertex < neighbor:  # Лексикографічне порівняння
                    edges.append(Edge(vertex, neighbor, weight))
        
        # Виконуємо алгоритми
        prim_edges, prim_weight = GraphAlgorithms.prim_mst(graph)
        kruskal_edges, kruskal_weight = GraphAlgorithms.kruskal_mst(vertices, edges)
        
        # Створюємо NetworkX граф
        G = nx.Graph()
        for vertex, neighbors in graph.items():
            for neighbor, weight in neighbors:
                G.add_edge(vertex, neighbor, weight=weight)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        pos = nx.spring_layout(G, seed=42)
        
        # Оригінальний граф
        nx.draw(G, pos, ax=ax1, with_labels=True, node_color='lightblue',
                node_size=800, font_size=10, font_weight='bold')
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax1, font_size=8)
        ax1.set_title('Оригінальний граф')
        
        # MST за алгоритмом Пріма
        prim_edge_list = [(edge.start, edge.end) for edge in prim_edges]
        nx.draw(G, pos, ax=ax2, with_labels=True, node_color='lightgreen',
                node_size=800, font_size=10, font_weight='bold', edge_color='lightgray')
        nx.draw_networkx_edges(G, pos, prim_edge_list, ax=ax2, edge_color='blue', width=3)
        ax2.set_title(f'MST (Алгоритм Пріма)\nВага: {prim_weight:.1f}')
        
        # MST за алгоритмом Крускала
        kruskal_edge_list = [(edge.start, edge.end) for edge in kruskal_edges]
        nx.draw(G, pos, ax=ax3, with_labels=True, node_color='lightcoral',
                node_size=800, font_size=10, font_weight='bold', edge_color='lightgray')
        nx.draw_networkx_edges(G, pos, kruskal_edge_list, ax=ax3, edge_color='red', width=3)
        ax3.set_title(f'MST (Алгоритм Крускала)\nВага: {kruskal_weight:.1f}')
        
        plt.tight_layout()
        plt.show()
        
        # Перевіряємо, чи дають однакові результати
        print(f"Вага MST (Прім): {prim_weight:.1f}")
        print(f"Вага MST (Крускал): {kruskal_weight:.1f}")
        print(f"Результати збігаються: {'Так' if abs(prim_weight - kruskal_weight) < 0.001 else 'Ні'}")
    
    @staticmethod
    def visualize_tsp_solution(distances: Dict[Tuple[str, str], float], start: str):
        """Візуалізує розв'язок TSP жадібною евристикою"""
        route, total_distance = TSPSolver.nearest_neighbor_tsp(distances, start)
        
        # Створюємо повний граф з містами
        cities = set()
        for city1, city2 in distances.keys():
            cities.add(city1)
            cities.add(city2)
        
        G = nx.complete_graph(list(cities))
        
        # Додаємо ваги ребер
        for (city1, city2), dist in distances.items():
            if G.has_edge(city1, city2):
                G[city1][city2]['weight'] = dist
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        pos = nx.spring_layout(G, seed=42)
        
        # Лівий графік: повний граф
        nx.draw(G, pos, ax=ax1, with_labels=True, node_color='lightblue',
                node_size=1000, font_size=12, font_weight='bold', alpha=0.7)
        ax1.set_title('Повний граф міст\n(Задача Комівояжера)')
        
        # Правий графік: знайдений маршрут
        nx.draw(G, pos, ax=ax2, with_labels=True, node_color='lightgreen',
                node_size=1000, font_size=12, font_weight='bold', alpha=0.3)
        
        # Виділяємо маршрут
        route_edges = []
        for i in range(len(route) - 1):
            route_edges.append((route[i], route[i + 1]))
        
        nx.draw_networkx_edges(G, pos, route_edges, ax=ax2, edge_color='red', width=3)
        
        # Додаємо порядок відвідування
        for i, city in enumerate(route[:-1]):  # Виключаємо повторення початкового міста
            x, y = pos[city]
            ax2.text(x, y+0.1, str(i+1), ha='center', va='center',
                    bbox={'boxstyle': 'circle,pad=0.3', 'facecolor': 'yellow'})
        
        ax2.set_title(f'Жадібний розв\'язок TSP\nМаршрут: {" → ".join(route)}\nДовжина: {total_distance:.1f}')
        
        plt.tight_layout()
        plt.show()
        
        return route, total_distance