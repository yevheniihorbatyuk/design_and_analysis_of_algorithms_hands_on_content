# src/utils/graph_generator.py

import networkx as nx
import numpy as np

class GraphGenerator:
    """Клас для генерації релевантних графів для різних доменів."""
    @staticmethod
    def create_city_road_network(n_intersections=50, connectivity=0.1, seed=42):
        np.random.seed(seed)
        G = nx.Graph()
        pos = {i: (np.random.uniform(0, 10), np.random.uniform(0, 10)) for i in range(n_intersections)}
        G.add_nodes_from(pos.keys())

        for i in range(n_intersections):
            for j in range(i + 1, n_intersections):
                if np.random.rand() < connectivity:
                    dist = np.sqrt((pos[i][0] - pos[j][0])**2 + (pos[i][1] - pos[j][1])**2)
                    G.add_edge(i, j, weight=dist * (1 + np.random.uniform(0, 0.3)))
        return G, pos

    @staticmethod
    def create_social_network(n_users=20, avg_friends=4, seed=42):
        G = nx.watts_strogatz_graph(n_users, avg_friends, 0.3, seed=seed)
        names = {i: f"User_{i}" for i in range(n_users)}
        return G, names
        
    @staticmethod
    def create_logistics_network(n_hubs=3, n_warehouses=10, n_stores=30, seed=42):
        np.random.seed(seed)
        G = nx.Graph()
        total_nodes = n_hubs + n_warehouses + n_stores
        pos = {i: (np.random.rand() * 100, np.random.rand() * 100) for i in range(total_nodes)}
        
        node_types = {}
        for i in range(n_hubs): node_types[i] = 'hub'
        for i in range(n_hubs, n_hubs + n_warehouses): node_types[i] = 'warehouse'
        for i in range(n_hubs + n_warehouses, total_nodes): node_types[i] = 'store'

        # Connect hubs
        for i in range(n_hubs):
            for j in range(i + 1, n_hubs):
                G.add_edge(i, j, weight=np.random.uniform(200, 500))

        # Connect warehouses to hubs
        for i in range(n_hubs, n_hubs + n_warehouses):
            hub = np.random.choice(n_hubs)
            G.add_edge(i, hub, weight=np.random.uniform(50, 150))

        # Connect stores to warehouses
        for i in range(n_hubs + n_warehouses, total_nodes):
            warehouse = np.random.choice(range(n_hubs, n_hubs + n_warehouses))
            G.add_edge(i, warehouse, weight=np.random.uniform(10, 80))
            
        return G, node_types, pos

    @staticmethod
    def create_dependency_graph_with_cycle(seed=42):
        np.random.seed(seed)
        G = nx.DiGraph()
        tasks = {0: "Core", 1: "Utils", 2: "API", 3: "Database", 4: "Auth", 5: "UI"}
        G.add_nodes_from(tasks.keys())
        dependencies = [(1,0), (2,1), (3,0), (4,2), (4,3), (5,2), (0,4)] # Cycle: Core -> Auth -> API -> Utils -> Core
        G.add_edges_from(dependencies)
        return G, tasks