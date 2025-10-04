# src/algorithms/shortest_path.py

import heapq
import time
from collections import deque
import numpy as np
import pandas as pd

class Dijkstra:
    """Реалізація алгоритму Дейкстри."""
    @staticmethod
    def find_path(graph, start_node, end_node):
        start_time = time.perf_counter()
        
        distances = {node: float('inf') for node in graph.nodes()}
        distances[start_node] = 0
        predecessors = {node: None for node in graph.nodes()}
        
        pq = [(0, start_node)]
        visited_nodes = set()

        while pq:
            current_distance, current_node = heapq.heappop(pq)
            
            if current_node in visited_nodes:
                continue
            
            visited_nodes.add(current_node)

            if current_node == end_node:
                break

            for neighbor in graph.neighbors(current_node):
                weight = graph[current_node][neighbor].get('weight', 1)
                distance = current_distance + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current_node
                    heapq.heappush(pq, (distance, neighbor))
        
        # Відновлення шляху
        path = []
        current = end_node
        while current is not None:
            path.append(current)
            current = predecessors[current]
        path.reverse()
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        return {
            'path': path if path and path[0] == start_node else [],
            'distance': distances[end_node],
            'visited_nodes': visited_nodes,
            'execution_time': execution_time,
            'algorithm': 'Dijkstra'
        }

class AStar:
    """Реалізація алгоритму A*."""
    @staticmethod
    def _euclidean_distance(pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    @staticmethod
    def _manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    @staticmethod
    def find_path(graph, start_node, end_node, positions, heuristic='euclidean'):
        start_time = time.perf_counter()

        if heuristic == 'manhattan':
            h_func = lambda n: AStar._manhattan_distance(positions.get(n), positions.get(end_node))
        else:
            h_func = lambda n: AStar._euclidean_distance(positions.get(n), positions.get(end_node))

        g_scores = {node: float('inf') for node in graph.nodes()}
        g_scores[start_node] = 0
        
        f_scores = {node: float('inf') for node in graph.nodes()}
        f_scores[start_node] = h_func(start_node)

        predecessors = {node: None for node in graph.nodes()}
        pq = [(f_scores[start_node], start_node)]
        visited_nodes = set()

        while pq:
            _, current_node = heapq.heappop(pq)
            
            if current_node in visited_nodes:
                continue

            visited_nodes.add(current_node)
            
            if current_node == end_node:
                break

            for neighbor in graph.neighbors(current_node):
                weight = graph[current_node][neighbor].get('weight', 1)
                tentative_g_score = g_scores[current_node] + weight

                if tentative_g_score < g_scores[neighbor]:
                    predecessors[neighbor] = current_node
                    g_scores[neighbor] = tentative_g_score
                    f_scores[neighbor] = tentative_g_score + h_func(neighbor)
                    heapq.heappush(pq, (f_scores[neighbor], neighbor))
        
        path = []
        current = end_node
        while current is not None:
            path.append(current)
            current = predecessors[current]
        path.reverse()

        execution_time = (time.perf_counter() - start_time) * 1000

        return {
            'path': path if path and path[0] == start_node else [],
            'distance': g_scores[end_node],
            'visited_nodes': visited_nodes,
            'execution_time': execution_time,
            'algorithm': 'A*'
        }

class BellmanFord:
    """Реалізація алгоритму Беллмана-Форда."""
    @staticmethod
    def find_path(graph, start_node, end_node):
        start_time = time.perf_counter()
        
        nodes = list(graph.nodes())
        edges = list(graph.edges(data=True))
        
        distances = {node: float('inf') for node in nodes}
        distances[start_node] = 0
        predecessors = {node: None for node in nodes}

        # Релаксація ребер V-1 разів
        for _ in range(len(nodes) - 1):
            for u, v, data in edges:
                weight = data.get('weight', 1)
                if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    predecessors[v] = u
        
        # Перевірка на від'ємні цикли
        has_negative_cycle = False
        for u, v, data in edges:
            weight = data.get('weight', 1)
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                has_negative_cycle = True
                break
        
        path = []
        if not has_negative_cycle:
            current = end_node
            while current is not None:
                path.append(current)
                current = predecessors.get(current)
            path.reverse()

        execution_time = (time.perf_counter() - start_time) * 1000

        return {
            'path': path if path and path[0] == start_node else [],
            'distance': distances.get(end_node, float('inf')),
            'visited_nodes': set(nodes), # Bellman-Ford effectively "visits" all nodes
            'execution_time': execution_time,
            'algorithm': 'Bellman-Ford',
            'has_negative_cycle': has_negative_cycle
        }

# class FloydWarshall:
#     """Реалізація алгоритму Флойда-Воршелла."""
#     @staticmethod
#     def find_all_paths(graph):
#         start_time = time.perf_counter()
        
#         nodes = list(graph.nodes())
#         node_map = {node: i for i, node in enumerate(nodes)}
#         num_nodes = len(nodes)
        
#         dist = np.full((num_nodes, num_nodes), np.inf)
#         next_node = np.full((num_nodes, num_nodes), -1, dtype=int)

#         for i in range(num_nodes):
#             dist[i, i] = 0

#         for u, v, data in graph.edges(data=True):
#             i, j = node_map[u], node_map[v]
#             weight = data.get('weight', 1)
#             dist[i, j] = weight
#             dist[j, i] = weight # For undirected graph
#             next_node[i, j] = j
#             next_node[j, i] = i
            
#         for k in range(num_nodes):
#             for i in range(num_nodes):
#                 for j in range(num_nodes):
#                     if dist[i, k] + dist[k, j] < dist[i, j]:
#                         dist[i, j] = dist[i, k] + dist[k, j]
#                         next_node[i, j] = next_node[i, k]
                        
#         execution_time = (time.perf_counter() - start_time) * 1000
        
#         return {
#             'distances': dist,
#             'next_nodes': next_node,
#             'node_map': node_map,
#             'execution_time': execution_time
#         }
    
class FloydWarshall:
    """Реалізація алгоритму Флойда-Воршелла."""
    @staticmethod
    def find_all_paths(graph):
        start_time = time.perf_counter()
        
        nodes = list(graph.nodes())
        node_map = {node: i for i, node in enumerate(nodes)}
        inv_node_map = {i: node for node, i in node_map.items()}
        num_nodes = len(nodes)
        
        dist = np.full((num_nodes, num_nodes), np.inf)
        
        for i in range(num_nodes):
            dist[i, i] = 0

        for u, v, data in graph.edges(data=True):
            i, j = node_map[u], node_map[v]
            weight = data.get('weight', 1)
            dist[i, j] = min(dist[i, j], weight)
            if not graph.is_directed():
                 dist[j, i] = min(dist[j, i], weight)
            
        for k in range(num_nodes):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
                        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        # Convert matrix back to a DataFrame for readability
        dist_df = pd.DataFrame(dist, index=nodes, columns=nodes)

        return {
            'distances_df': dist_df,
            'execution_time': execution_time
        }

class BidirectionalSearch:
    """Реалізація двонаправленого пошуку (на основі BFS)."""
    @staticmethod
    def find_path(graph, start_node, end_node):
        start_time = time.perf_counter()

        if start_node == end_node:
            return {'path': [start_node], 'distance': 0, 'visited_nodes': {start_node}, 'execution_time': 0, 'algorithm': 'Bidirectional'}

        q_fwd, q_bwd = deque([start_node]), deque([end_node])
        visited_fwd, visited_bwd = {start_node: None}, {end_node: None}
        meeting_point = None

        while q_fwd and q_bwd:
            # Forward search step
            curr_fwd = q_fwd.popleft()
            for neighbor in graph.neighbors(curr_fwd):
                if neighbor not in visited_fwd:
                    visited_fwd[neighbor] = curr_fwd
                    q_fwd.append(neighbor)
                    if neighbor in visited_bwd:
                        meeting_point = neighbor
                        break
            if meeting_point: break

            # Backward search step
            curr_bwd = q_bwd.popleft()
            for neighbor in graph.neighbors(curr_bwd):
                if neighbor not in visited_bwd:
                    visited_bwd[neighbor] = curr_bwd
                    q_bwd.append(neighbor)
                    if neighbor in visited_fwd:
                        meeting_point = neighbor
                        break
            if meeting_point: break
        
        path = []
        if meeting_point:
            path_fwd = []
            curr = meeting_point
            while curr is not None:
                path_fwd.append(curr)
                curr = visited_fwd[curr]
            
            path_bwd = []
            curr = visited_bwd[meeting_point]
            while curr is not None:
                path_bwd.append(curr)
                curr = visited_bwd[curr]
            
            path = list(reversed(path_fwd)) + path_bwd

        execution_time = (time.perf_counter() - start_time) * 1000
        total_visited = set(visited_fwd.keys()) | set(visited_bwd.keys())

        # Note: Distance calculation is for unweighted graphs here
        return {
            'path': path,
            'distance': len(path) - 1 if path else float('inf'),
            'visited_nodes': total_visited,
            'execution_time': execution_time,
            'algorithm': 'Bidirectional'
        }