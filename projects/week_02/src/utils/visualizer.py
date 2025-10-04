# src/utils/visualizer.py

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class PathVisualizer:
    """Клас для візуалізації шляхів та графів."""
    @staticmethod
    def draw_path(graph, path, title, pos=None):
        if pos is None:
            pos = nx.spring_layout(graph, seed=42)
        
        plt.figure(figsize=(12, 8))
        
        # Draw base graph
        nx.draw_networkx_nodes(graph, pos, node_color='lightblue', node_size=500)
        nx.draw_networkx_edges(graph, pos, alpha=0.5, edge_color='gray')
        nx.draw_networkx_labels(graph, pos)
        
        # Highlight path
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_nodes(graph, pos, nodelist=path, node_color='tomato', node_size=700)
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='tomato', width=3)
        
        edge_labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        return plt.gcf()

    @staticmethod
    def draw_logistics_network(graph, node_types, pos, path, title):
        plt.figure(figsize=(16, 12))
        
        color_map = {'hub': 'red', 'warehouse': 'orange', 'store': 'lightblue'}
        node_colors = [color_map[node_types[node]] for node in graph.nodes()]
        
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=400)
        nx.draw_networkx_edges(graph, pos, alpha=0.3, edge_color='gray')
        
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_nodes(graph, pos, nodelist=path, node_color='springgreen', node_size=600, edgecolors='black')
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='springgreen', width=4)
        
        plt.title(title, fontsize=18, fontweight='bold')
        plt.legend(handles=[plt.Line2D([0], [0], color=c, marker='o', linestyle='', label=l) for l, c in color_map.items()])
        plt.axis('off')
        return plt.gcf()

    @staticmethod
    def draw_visited_nodes_comparison(graph, pos, results, title):
        num_algos = len(results)
        fig, axes = plt.subplots(1, num_algos, figsize=(8 * num_algos, 7))
        if num_algos == 1:
            axes = [axes]

        for ax, result in zip(axes, results):
            visited = set(result.get('visited_nodes', []))
            algo_name = result.get('algorithm', 'Algorithm')

            unvisited_nodes = set(graph.nodes()) - visited

            nx.draw_networkx_nodes(graph, pos, nodelist=list(unvisited_nodes),
                                node_color='lightgray', node_size=100, ax=ax)
            nx.draw_networkx_nodes(graph, pos, nodelist=list(visited),
                                node_color='orange', node_size=150, alpha=0.8, ax=ax)
            nx.draw_networkx_edges(graph, pos, alpha=0.2, edge_color='gray', ax=ax)

            path = result.get('path', [])
            if path:
                start_node = path[0]
                end_node = path[-1]
                nx.draw_networkx_nodes(graph, pos, nodelist=[start_node],
                                    node_color='green', node_size=300, node_shape='s', ax=ax)
                nx.draw_networkx_nodes(graph, pos, nodelist=[end_node],
                                    node_color='red', node_size=300, node_shape='s', ax=ax)

            ax.set_title(f"{algo_name}\n(Відвідано: {len(visited)} вузлів)", fontsize=14)
            ax.axis('off')

        fig.suptitle(title, fontsize=18, fontweight='bold')
        return fig

        
class TraversalVisualizer:
    @staticmethod
    def draw_bfs_levels(graph, start_node, distances, labels, title):
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(graph, seed=42)
        
        max_dist = max(distances.values())
        colors = plt.cm.get_cmap('viridis', max_dist + 1)
        node_colors = [colors(distances[node]) for node in graph.nodes()]
        
        nx.draw(graph, pos, labels=labels, with_labels=True, node_color=node_colors, node_size=2000, font_size=10, font_color='white', edge_color='gray')
        
        plt.title(title, fontsize=18, fontweight='bold')
        return plt.gcf()
        
    @staticmethod
    def draw_dfs_order(graph, order, labels, title):
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(graph, seed=42)
        
        colors = plt.cm.get_cmap('plasma', len(order))
        order_map = {node: i for i, node in enumerate(order)}
        node_colors = [colors(order_map[node]) for node in graph.nodes()]
        
        nx.draw(graph, pos, labels=labels, with_labels=True, node_color=node_colors, node_size=2000, font_size=10, font_color='white', edge_color='gray')
        
        plt.title(title, fontsize=18, fontweight='bold')
        return plt.gcf()

    @staticmethod
    def draw_cycle(graph, cycle, labels, title):
        plt.figure(figsize=(12, 8))
        pos = nx.circular_layout(graph)

        nx.draw_networkx_nodes(graph, pos, node_color='lightblue', node_size=2500)
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=12, font_weight='bold')
        nx.draw_networkx_edges(graph, pos, alpha=0.3, edge_color='gray', arrows=True, node_size=2500)

        # 🔧 Нормалізація: якщо у cycle не ID, а ярлики — конвертуємо назад у ID
        if cycle:
            if cycle and cycle[0] not in graph:  # елемент не у G.nodes()
                label_to_node = {lbl: n for n, lbl in labels.items()}
                try:
                    cycle = [label_to_node[lbl] for lbl in cycle]
                except KeyError as e:
                    raise ValueError(f"Label {e.args[0]!r} відсутній у labels → неможливо побудувати цикл") from None

            # будуємо ребра циклу (+ замикання)
            cycle_edges = list(zip(cycle, cycle[1:]))
            if len(cycle) > 2:
                cycle_edges.append((cycle[-1], cycle[0]))

            nx.draw_networkx_edges(
                graph, pos,
                edgelist=cycle_edges,
                edge_color='red', width=3, arrows=True, node_size=2500
            )

        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        return plt.gcf()


class MstVisualizer:
    @staticmethod
    def draw_mst(graph, mst_edges, title, pos=None):
        if pos is None:
            pos = nx.spring_layout(graph, seed=42)
        
        plt.figure(figsize=(14, 10))
        
        # Draw all edges faintly
        nx.draw_networkx_edges(graph, pos, alpha=0.2, edge_color='gray')
        
        # Draw MST edges boldly
        nx.draw_networkx_edges(graph, pos, edgelist=[(u, v) for u, v, w in mst_edges], edge_color='red', width=2.5)
        
        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, node_color='skyblue', node_size=700)
        
        # Draw labels
        labels = {node: str(node) for node in graph.nodes()}
        nx.draw_networkx_labels(graph, pos, labels=labels, font_color='black')
        
        edge_labels = {(u, v): f"{w['weight']:.1f}" for u, v, w in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, alpha=0.7)
        
        plt.title(title, fontsize=18, fontweight='bold')
        plt.axis('off')
        return plt.gcf()
    
class AnalysisVisualizer:
    @staticmethod
    def plot_distance_matrix(df, title):
        plt.figure(figsize=(10, 8))
        sns.heatmap(df, annot=True, fmt=".1f", cmap="viridis_r", linewidths=.5)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel("Кінцева вершина")
        plt.ylabel("Початкова вершина")
        return plt.gcf()
    

class TreeVisualizer:
    """Клас для візуалізації деревоподібних структур."""
    @staticmethod
    def _get_tree_graph(node, graph=None):
        if graph is None:
            graph = nx.DiGraph()
        if node is not None:
            if node.left:
                graph.add_edge(node.key, node.left.key)
                TreeVisualizer._get_tree_graph(node.left, graph)
            if node.right:
                graph.add_edge(node.key, node.right.key)
                TreeVisualizer._get_tree_graph(node.right, graph)
        return graph

    @staticmethod
    def draw_tree(tree_root, title, figsize=(10, 6)):
        if not tree_root:
            print(f"Попередження: Дерево '{title}' порожнє.")
            return

        plt.figure(figsize=figsize)
        g = TreeVisualizer._get_tree_graph(tree_root)
        
        # Використовуємо graphviz_layout для ієрархічного вигляду, якщо доступно
        try:
            pos = nx.nx_agraph.graphviz_layout(g, prog='dot')
        except ImportError:
            print("Попередження: PyGraphviz не встановлено. Використовується менш ієрархічний layout.")
            pos = nx.spring_layout(g, iterations=100)

        nx.draw(g, pos, with_labels=True, node_color='skyblue', node_size=1500,
                edge_color='gray', font_size=10, font_weight='bold',
                arrows=False)
        plt.title(title, fontsize=16, fontweight='bold')
        return plt.gcf()
    



from matplotlib.patches import Rectangle

    
class GeospatialVisualizer:
    """Клас для візуалізації геопросторових структур."""
    @staticmethod
    def draw_kd_tree_search(points, query_point, nearest_neighbor, title):
        plt.figure(figsize=(10, 10))
        points_arr = np.array(points)
        
        plt.scatter(points_arr[:, 0], points_arr[:, 1], c='lightblue', label='Інші точки')
        plt.scatter(query_point[0], query_point[1], c='red', s=150, marker='X', label='Точка запиту')
        if nearest_neighbor:
            nn_arr = np.array(nearest_neighbor)
            plt.scatter(nn_arr[0], nn_arr[1], c='green', s=150, marker='*', label='Найближчий сусід')
            # Draw line
            plt.plot([query_point[0], nn_arr[0]], [query_point[1], nn_arr[1]], 'r--')

        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel("X координата")
        plt.ylabel("Y координата")
        plt.legend()
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        return plt.gcf()

    @staticmethod
    def draw_quadtree(quadtree_node, ax):
        x, y, w, h = quadtree_node.boundary
        rect = Rectangle((x, y), w, h, linewidth=0.6, edgecolor='gray', facecolor='none', alpha=0.8)
        ax.add_patch(rect)
        if quadtree_node.divided:
            for child in quadtree_node.children:
                GeospatialVisualizer.draw_quadtree(child, ax)

    @staticmethod
    def draw_quadtree_range_search(quadtree, points, query_boundary, found_points, title):
        fig, ax = plt.subplots(figsize=(10, 10))
        points_arr = np.array(points)
        found_arr = np.array(found_points) if found_points else np.empty((0,2))
        
        # Draw all points
        ax.scatter(points_arr[:, 0], points_arr[:, 1], c='lightgray', s=10, label='Всі об\'єкти')
        
        # Draw Quadtree structure
        GeospatialVisualizer.draw_quadtree(quadtree.root, ax)
        
        # Draw query range
        qx, qy, qw, qh = query_boundary
        query_rect = Rectangle((qx, qy), qw, qh, linewidth=2, edgecolor='red', facecolor='red', alpha=0.2, label='Область запиту')
        ax.add_patch(query_rect)
        
        # Highlight found points
        if found_points:
            ax.scatter(found_arr[:, 0], found_arr[:, 1], c='springgreen', s=50, edgecolors='black', label='Знайдені об\'єкти')
            
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        ax.set_aspect('equal', adjustable='box')
        return fig