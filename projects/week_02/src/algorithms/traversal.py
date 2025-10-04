# src/algorithms/traversal.py

from collections import deque

import networkx as nx
import numpy as np

def find_negative_cycle_bf(G: nx.DiGraph, weight: str = "weight", eps: float = 1e-12):
    """
    Повертає список вершин від'ємного циклу (у напрямку дуг), або None якщо циклу немає.
    Працює для довільного старту (через "супер-джерело": dist=0 для всіх).
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return None

    # 0 для всіх — дозволяє охопити всі компоненти
    dist = {u: 0.0 for u in nodes}
    pred = {u: None for u in nodes}

    # Готуємо список ребер
    edges = []
    for u, v, data in G.edges(data=True):
        w = data.get(weight, 0.0)
        edges.append((u, v, float(w)))

    # |V|-1 релаксацій
    for _ in range(max(n - 1, 0)):
        updated = False
        for u, v, w in edges:
            if dist[u] + w < dist[v] - eps:
                dist[v] = dist[u] + w
                pred[v] = u
                updated = True
        if not updated:
            break

    # Перевірка на від'ємний цикл
    x = None
    for u, v, w in edges:
        if dist[u] + w < dist[v] - eps:
            pred[v] = u
            x = v
            break

    if x is None:
        return None  # циклу нема

    # Крокаємо n разів по предках, щоб гарантовано потрапити всередину циклу
    for _ in range(n):
        x = pred[x]

    # Відновлюємо цикл, рухаючись по предках, доки не повернемось у початкову вершину
    cycle = [x]
    cur = pred[x]
    while cur is not None and cur != x:
        cycle.append(cur)
        cur = pred[cur]
        if len(cycle) > n + 5:  # запобігання рідкісним патологіям
            break
    cycle.append(x)

    # Тепер cycle має напрямок "назад" уздовж pred; розвернемо для напрямку дуг
    cycle.reverse()

    # (Опційно) прибираємо дубль останньої вершини для зручності подальшого використання
    # зараз вигляд [v0, v1, ..., vk, v0]; якщо хочеш без замикання, розкоментуй:
    # cycle = cycle[:-1]

    return cycle


class BFS:
    """Реалізація Пошуку в ширину (BFS)."""
    def __init__(self, graph):
        self.graph = graph

    def traverse(self, start_node):
        visited = {start_node}
        queue = deque([start_node])
        order = []
        distances = {start_node: 0}
        parent = {start_node: None}

        while queue:
            current = queue.popleft()
            order.append(current)

            for neighbor in self.graph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    distances[neighbor] = distances[current] + 1
                    parent[neighbor] = current
        
        return {
            'order': order,
            'distances': distances,
            'parent': parent
        }

class DFS:
    """Реалізація Пошуку в глибину (DFS)."""
    def __init__(self, graph):
        self.graph = graph
        self.visited = set()
        self.order = []
        self.parent = {}

    def traverse(self, start_node):
        self.visited.clear()
        self.order.clear()
        self.parent.clear()
        self._dfs_recursive(start_node, None)
        return {'order': self.order, 'parent': self.parent}

    def _dfs_recursive(self, node, p):
        self.visited.add(node)
        self.order.append(node)
        self.parent[node] = p
        for neighbor in self.graph.neighbors(node):
            if neighbor not in self.visited:
                self._dfs_recursive(neighbor, node)

    def find_cycle(self):
        visited = set()
        recursion_stack = set()
        parent = {}

        for node in self.graph.nodes():
            if node not in visited:
                cycle = self._find_cycle_util(node, visited, recursion_stack, parent)
                if cycle:
                    return cycle
        return None
    
    def _find_cycle_util(self, node, visited, recursion_stack, parent):
        visited.add(node)
        recursion_stack.add(node)
        
        for neighbor in self.graph.neighbors(node):
            if neighbor not in visited:
                parent[neighbor] = node
                found_cycle = self._find_cycle_util(neighbor, visited, recursion_stack, parent)
                if found_cycle:
                    return found_cycle
            elif neighbor in recursion_stack:
                # Cycle detected
                cycle = [neighbor, node]
                curr = node
                while parent.get(curr) != neighbor:
                    curr = parent.get(curr)
                    if curr is None: break
                    cycle.append(curr)
                cycle.append(neighbor)
                return list(reversed(cycle))
        
        recursion_stack.remove(node)
        return None
    
