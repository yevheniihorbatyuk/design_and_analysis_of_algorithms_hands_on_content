# src/algorithms/mst.py

import heapq
from ..data_structures.union_find import UnionFind

class Kruskal:
    """Реалізація алгоритму Краскала."""
    @staticmethod
    def find_mst(graph):
        edges = sorted(graph.edges(data=True), key=lambda t: t[2].get('weight', 1))
        
        node_map = {node: i for i, node in enumerate(graph.nodes())}
        uf = UnionFind(len(graph.nodes()))
        
        mst_edges = []
        total_weight = 0

        for u, v, data in edges:
            if uf.union(node_map[u], node_map[v]):
                weight = data.get('weight', 1)
                mst_edges.append((u, v, weight))
                total_weight += weight
                if len(mst_edges) == len(graph.nodes()) - 1:
                    break
        
        return {'edges': mst_edges, 'total_weight': total_weight}

class Prim:
    """Реалізація алгоритму Пріма."""
    @staticmethod
    def find_mst(graph, start_node=None):
        if start_node is None:
            start_node = list(graph.nodes())[0]

        visited = {start_node}
        pq = []
        for neighbor in graph.neighbors(start_node):
            weight = graph[start_node][neighbor].get('weight', 1)
            heapq.heappush(pq, (weight, start_node, neighbor))

        mst_edges = []
        total_weight = 0

        while pq and len(visited) < len(graph.nodes()):
            weight, u, v = heapq.heappop(pq)
            if v in visited:
                continue

            visited.add(v)
            mst_edges.append((u, v, weight))
            total_weight += weight

            for neighbor in graph.neighbors(v):
                if neighbor not in visited:
                    new_weight = graph[v][neighbor].get('weight', 1)
                    heapq.heappush(pq, (new_weight, v, neighbor))
        
        return {'edges': mst_edges, 'total_weight': total_weight}