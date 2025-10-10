# src/algorithms/bsp_pagerank.py

import multiprocessing as mp
from multiprocessing import Manager, Barrier
from typing import Dict, List, Tuple, Set
import numpy as np
import time
from dataclasses import dataclass
import matplotlib.pyplot as plt
import networkx as nx

@dataclass
class BSPConfig:
    """Конфігурація BSP моделі."""
    num_workers: int
    max_supersteps: int = 20
    damping_factor: float = 0.85
    convergence_threshold: float = 1e-6

class PageNode:
    """Вузол графа для PageRank."""
    def __init__(self, node_id: int):
        self.node_id = node_id
        self.rank = 1.0
        self.new_rank = 0.0
        self.outgoing_edges: List[int] = []
        self.incoming_edges: List[int] = []
    
    def add_outgoing(self, target: int):
        if target not in self.outgoing_edges:
            self.outgoing_edges.append(target)
    
    def add_incoming(self, source: int):
        if source not in self.incoming_edges:
            self.incoming_edges.append(source)

# ============================================================================
# BSP Worker Process
# ============================================================================

def bsp_worker(
    worker_id: int,
    nodes_partition: List[int],
    graph: Dict[int, PageNode],
    config: BSPConfig,
    barrier: Barrier,
    shared_ranks: Dict,
    shared_messages: Dict,
    convergence_flag: mp.Value
):
    """
    BSP Worker процес. Виконує супер-кроки BSP:
    1. LOCAL COMPUTATION - обчислення нового рангу
    2. COMMUNICATION - відправка повідомлень сусідам
    3. BARRIER SYNCHRONIZATION - синхронізація всіх воркерів
    """
    
    damping = config.damping_factor
    n_nodes = len(graph)
    
    print(f"  🔷 Worker {worker_id}: відповідає за {len(nodes_partition)} вузлів: {nodes_partition[:5]}...")
    
    for superstep in range(config.max_supersteps):
        # ====================================================================
        # ФАЗА 1: LOCAL COMPUTATION
        # ====================================================================
        # Кожен воркер обчислює нові ранги для своїх вузлів
        local_diff = 0.0
        
        for node_id in nodes_partition:
            node = graph[node_id]
            
            # Отримуємо повідомлення від сусідів (з попереднього superstep)
            incoming_rank_sum = 0.0
            if node_id in shared_messages:
                incoming_rank_sum = sum(shared_messages[node_id])
                shared_messages[node_id] = []  # Очищаємо повідомлення
            
            # PageRank формула: PR(i) = (1-d)/N + d * Σ(PR(j)/L(j))
            new_rank = (1 - damping) / n_nodes + damping * incoming_rank_sum
            
            # Обчислюємо зміну для перевірки конвергенції
            local_diff += abs(new_rank - node.rank)
            
            node.new_rank = new_rank
        
        # ====================================================================
        # ФАЗА 2: COMMUNICATION
        # ====================================================================
        # Відправляємо повідомлення сусідам (про наш новий ранг)
        for node_id in nodes_partition:
            node = graph[node_id]
            
            # Оновлюємо ранг після обчислень
            node.rank = node.new_rank
            shared_ranks[node_id] = node.rank
            
            # Відправляємо частину нашого рангу всім вихідним сусідам
            if len(node.outgoing_edges) > 0:
                rank_to_send = node.rank / len(node.outgoing_edges)
                for target_id in node.outgoing_edges:
                    if target_id not in shared_messages:
                        shared_messages[target_id] = []
                    shared_messages[target_id].append(rank_to_send)
        
        # ====================================================================
        # ФАЗА 3: BARRIER SYNCHRONIZATION
        # ====================================================================
        # Всі воркери чекають один одного перед наступним superstep
        barrier.wait()
        
        # Тільки worker 0 перевіряє конвергенцію
        if worker_id == 0:
            # Обчислюємо глобальну різницю
            total_diff = sum(abs(shared_ranks.get(nid, 0) - graph[nid].new_rank) 
                           for nid in graph.keys())
            
            if total_diff < config.convergence_threshold:
                convergence_flag.value = 1
                print(f"\n  ✅ Конвергенція досягнута на superstep {superstep + 1}")
        
        # Синхронізація для перевірки конвергенції
        barrier.wait()
        
        if convergence_flag.value == 1:
            break
        
        if worker_id == 0 and (superstep + 1) % 5 == 0:
            print(f"  📊 Superstep {superstep + 1} завершено")
    
    if worker_id == 0:
        print(f"  🏁 Worker {worker_id}: BSP обчислення завершено")

# ============================================================================
# BSP PageRank Engine
# ============================================================================

class BSPPageRank:
    """BSP реалізація PageRank алгоритму."""
    
    def __init__(self, config: BSPConfig):
        self.config = config
        self.graph: Dict[int, PageNode] = {}
    
    def add_edge(self, source: int, target: int):
        """Додає ребро до графа."""
        if source not in self.graph:
            self.graph[source] = PageNode(source)
        if target not in self.graph:
            self.graph[target] = PageNode(target)
        
        self.graph[source].add_outgoing(target)
        self.graph[target].add_incoming(source)
    
    def partition_nodes(self) -> List[List[int]]:
        """Розбиває вузли на партиції для воркерів."""
        nodes = list(self.graph.keys())
        nodes_per_worker = len(nodes) // self.config.num_workers
        
        partitions = []
        for i in range(self.config.num_workers):
            start = i * nodes_per_worker
            end = start + nodes_per_worker if i < self.config.num_workers - 1 else len(nodes)
            partitions.append(nodes[start:end])
        
        return partitions
    
    def run(self) -> Tuple[Dict[int, float], float]:
        """Запускає BSP PageRank обчислення."""
        print(f"\n{'='*80}")
        print(f"🚀 ЗАПУСК BSP PAGERANK")
        print(f"{'='*80}")
        print(f"  Вузлів: {len(self.graph)}")
        print(f"  Ребер: {sum(len(node.outgoing_edges) for node in self.graph.values())}")
        print(f"  Воркерів: {self.config.num_workers}")
        print(f"  Макс. supersteps: {self.config.max_supersteps}")
        print(f"{'='*80}\n")
        
        start_time = time.perf_counter()
        
        # Розбиваємо вузли на партиції
        partitions = self.partition_nodes()
        
        # Створюємо shared objects для BSP
        manager = Manager()
        shared_ranks = manager.dict({nid: node.rank for nid, node in self.graph.items()})
        shared_messages = manager.dict()
        convergence_flag = mp.Value('i', 0)
        barrier = Barrier(self.config.num_workers)
        
        # Запускаємо воркерів
        processes = []
        for worker_id in range(self.config.num_workers):
            p = mp.Process(
                target=bsp_worker,
                args=(
                    worker_id,
                    partitions[worker_id],
                    self.graph,
                    self.config,
                    barrier,
                    shared_ranks,
                    shared_messages,
                    convergence_flag
                )
            )
            p.start()
            processes.append(p)
        
        # Чекаємо завершення всіх воркерів
        for p in processes:
            p.join()
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Збираємо результати
        final_ranks = dict(shared_ranks)
        
        print(f"\n{'='*80}")
        print(f"✅ BSP PAGERANK ЗАВЕРШЕНО")
        print(f"  Загальний час: {total_time:.2f} мс")
        print(f"{'='*80}\n")
        
        return final_ranks, total_time

# ============================================================================
# Допоміжні функції для створення тестових графів
# ============================================================================

def create_web_graph(num_pages: int, edge_probability: float = 0.1) -> BSPPageRank:
    """Створює випадковий веб-граф."""
    config = BSPConfig(num_workers=mp.cpu_count())
    pagerank = BSPPageRank(config)
    
    # Додаємо випадкові посилання між сторінками
    np.random.seed(42)
    for i in range(num_pages):
        for j in range(num_pages):
            if i != j and np.random.random() < edge_probability:
                pagerank.add_edge(i, j)
    
    return pagerank

def create_hub_and_spoke_graph(num_hubs: int = 5, spokes_per_hub: int = 10) -> BSPPageRank:
    """Створює граф з хабами і спицями (класичний для PageRank)."""
    config = BSPConfig(num_workers=mp.cpu_count())
    pagerank = BSPPageRank(config)
    
    node_id = 0
    for hub_idx in range(num_hubs):
        hub_id = node_id
        node_id += 1
        
        # Створюємо спиці навколо хаба
        spoke_ids = []
        for _ in range(spokes_per_hub):
            spoke_id = node_id
            spoke_ids.append(spoke_id)
            node_id += 1
            
            # Спиця посилається на хаб
            pagerank.add_edge(spoke_id, hub_id)
        
        # Хаби посилаються один на одного
        for other_hub in range(num_hubs):
            if hub_idx != other_hub:
                pagerank.add_edge(hub_id, other_hub)
        
        # Деякі спиці посилаються один на одного
        for i in range(len(spoke_ids)):
            if i + 1 < len(spoke_ids):
                pagerank.add_edge(spoke_ids[i], spoke_ids[i + 1])
    
    return pagerank

def create_social_network_graph(num_users: int = 100) -> BSPPageRank:
    """Створює граф соціальної мережі з деякими впливовими користувачами."""
    config = BSPConfig(num_workers=mp.cpu_count())
    pagerank = BSPPageRank(config)
    
    np.random.seed(42)
    
    # Створюємо впливових користувачів (10% від загальної кількості)
    num_influencers = max(1, num_users // 10)
    influencers = list(range(num_influencers))
    
    # Звичайні користувачі підписані на впливових
    for user in range(num_influencers, num_users):
        # Підписка на 2-5 впливових
        num_follows = np.random.randint(2, 6)
        followed_influencers = np.random.choice(influencers, size=min(num_follows, len(influencers)), replace=False)
        for influencer in followed_influencers:
            pagerank.add_edge(user, influencer)
        
        # Підписка на 1-3 інших звичайних користувачів
        num_friend_follows = np.random.randint(1, 4)
        for _ in range(num_friend_follows):
            friend = np.random.randint(num_influencers, num_users)
            if friend != user:
                pagerank.add_edge(user, friend)
    
    # Впливові також підписані один на одного
    for i in influencers:
        for j in influencers:
            if i != j and np.random.random() < 0.7:
                pagerank.add_edge(i, j)
    
    return pagerank

# ============================================================================
# Візуалізація результатів
# ============================================================================

def visualize_pagerank(pagerank: BSPPageRank, ranks: Dict[int, float], title: str = "PageRank Results"):
    """Візуалізує граф з PageRank результатами."""
    
    # Створюємо NetworkX граф
    G = nx.DiGraph()
    for node_id, node in pagerank.graph.items():
        G.add_node(node_id)
        for target in node.outgoing_edges:
            G.add_edge(node_id, target)
    
    # Налаштування візуалізації
    plt.figure(figsize=(16, 10))
    
    # Позиції вузлів
    if len(G.nodes()) <= 50:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    else:
        pos = nx.spring_layout(G, k=1, iterations=30, seed=42)
    
    # Розміри вузлів пропорційні PageRank
    node_sizes = [ranks[node] * 10000 for node in G.nodes()]
    
    # Кольори вузлів
    node_colors = [ranks[node] for node in G.nodes()]
    
    # Малюємо граф
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                          cmap=plt.cm.YlOrRd, alpha=0.8, edgecolors='black', linewidths=2)
    
    nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, arrowsize=20,
                          edge_color='gray', width=1.5, arrowstyle='->')
    
    # Підписи для топ-10 вузлів
    top_nodes = sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:10]
    labels = {node_id: f"{node_id}\n{rank:.3f}" for node_id, rank in top_nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    # Додаємо colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, 
                               norm=plt.Normalize(vmin=min(ranks.values()), 
                                                vmax=max(ranks.values())))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.046, pad=0.04)
    cbar.set_label('PageRank Score', fontweight='bold')
    
    plt.savefig(f'pagerank_{title.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Графік збережено: pagerank_{title.lower().replace(' ', '_')}.png")
    plt.show()

def print_top_pages(ranks: Dict[int, float], top_n: int = 10):
    """Виводить топ-N сторінок за PageRank."""
    sorted_ranks = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'='*60}")
    print(f"🏆 ТОП-{top_n} СТОРІНОК ЗА PAGERANK")
    print(f"{'='*60}")
    print(f"{'Позиція':<10} {'ID вузла':<15} {'PageRank':<15} {'%':<10}")
    print(f"{'-'*60}")
    
    total_rank = sum(ranks.values())
    for idx, (node_id, rank) in enumerate(sorted_ranks[:top_n], 1):
        percentage = (rank / total_rank) * 100
        medal = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else "  "
        print(f"{medal} #{idx:<7} {node_id:<15} {rank:<15.6f} {percentage:<10.2f}%")
    
    print(f"{'='*60}\n")