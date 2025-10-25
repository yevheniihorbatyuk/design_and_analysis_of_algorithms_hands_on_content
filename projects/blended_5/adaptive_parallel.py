from enum import Enum
from typing import List, Dict, Tuple, Set, Optional
import networkx as nx
import numpy as np
from dataclasses import dataclass, field

class MigrationTopology(Enum):
    """Типи топологій для міграції."""
    RING = "ring"  # Кільцева топологія
    FULLY_CONNECTED = "fully_connected"  # Повністю зв'язана
    RANDOM = "random"  # Випадкові зв'язки
    HIERARCHICAL = "hierarchical"  # Ієрархічна структура

@dataclass
class AdaptiveIslandConfig(IslandConfig):
    """Розширена конфігурація з адаптивними параметрами."""
    min_mutation_rate: float = 0.01
    max_mutation_rate: float = 0.2
    min_crossover_rate: float = 0.6
    max_crossover_rate: float = 0.95
    adaptation_rate: float = 0.1  # швидкість адаптації параметрів
    diversity_threshold: float = 0.1  # поріг різноманітності популяції
    
    # Історія параметрів та продуктивності
    parameter_history: Dict[str, List[float]] = field(default_factory=lambda: {
        'mutation_rate': [],
        'crossover_rate': [],
        'local_search_freq': [],
        'diversity': [],
        'improvement_rate': []
    })

class AdaptiveIsland(Island):
    """Острів з адаптивними механізмами."""
    
    def __init__(self, island_id: int, config: AdaptiveIslandConfig,
                 initial_solution: Solution, migration_queue: Queue,
                 topology: MigrationTopology):
        super().__init__(island_id, config, initial_solution, migration_queue)
        self.config = config  # перевизначаємо як AdaptiveIslandConfig
        self.topology = topology
        self.neighbors: Set[int] = set()  # сусідні острови
        self.improvement_history: List[float] = []
        
    def calculate_diversity(self) -> float:
        """Обчислює різноманітність популяції."""
        if not self.population:
            return 0.0
            
        # Обчислюємо попарні відстані між хромосомами
        distances = []
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                distance = sum(1 for a, b in zip(self.population[i], 
                                               self.population[j]) if a != b)
                distances.append(distance)
        
        return np.mean(distances) / len(self.population[0]) if distances else 0.0
    
    def adapt_parameters(self):
        """Адаптує параметри на основі продуктивності та різноманітності."""
        diversity = self.calculate_diversity()
        
        # Зберігаємо поточні значення параметрів
        self.config.parameter_history['diversity'].append(diversity)
        self.config.parameter_history['mutation_rate'].append(
            self.config.mutation_rate)
        self.config.parameter_history['crossover_rate'].append(
            self.config.crossover_rate)
        self.config.parameter_history['local_search_freq'].append(
            self.config.local_search_freq)
        
        # Адаптуємо mutation_rate на основі різноманітності
        if diversity < self.config.diversity_threshold:
            # Збільшуємо mutation_rate для підвищення різноманітності
            self.config.mutation_rate = min(
                self.config.max_mutation_rate,
                self.config.mutation_rate * (1 + self.config.adaptation_rate)
            )
        else:
            # Зменшуємо mutation_rate для кращої збіжності
            self.config.mutation_rate = max(
                self.config.min_mutation_rate,
                self.config.mutation_rate * (1 - self.config.adaptation_rate)
            )
        
        # Адаптуємо crossover_rate на основі покращень
        if self.improvement_history and self.improvement_history[-1] > 0:
            # Якщо є покращення, збільшуємо crossover_rate
            self.config.crossover_rate = min(
                self.config.max_crossover_rate,
                self.config.crossover_rate * (1 + self.config.adaptation_rate)
            )
        else:
            # Якщо немає покращень, зменшуємо crossover_rate
            self.config.crossover_rate = max(
                self.config.min_crossover_rate,
                self.config.crossover_rate * (1 - self.config.adaptation_rate)
            )
        
        # Адаптуємо частоту локального пошуку
        if self.improvement_history and self.improvement_history[-1] > 0:
            # Якщо є покращення, збільшуємо частоту
            self.config.local_search_freq = min(
                0.5,  # максимальна частота
                self.config.local_search_freq * (1 + self.config.adaptation_rate)
            )
        else:
            # Якщо немає покращень, зменшуємо частоту
            self.config.local_search_freq = max(
                0.05,  # мінімальна частота
                self.config.local_search_freq * (1 - self.config.adaptation_rate)
            )
    
    def migrate_solutions(self, evaluated_population: List[Tuple[List[int], float]]):
        """Адаптивна міграція розв'язків."""
        if self.generation % self.config.migration_freq == 0:
            # Визначаємо розмір міграції на основі різноманітності
            diversity = self.calculate_diversity()
            migration_size = max(1, int(
                self.config.migration_size * (1 - diversity)
            ))
            
            best_solutions = evaluated_population[:migration_size]
            
            # Відправляємо розв'язки тільки сусідам
            for neighbor_id in self.neighbors:
                self.migration_queue.put((self.id, neighbor_id, best_solutions))
    
    def receive_migrants(self) -> List[List[int]]:
        """Отримує мігрантів від сусідів."""
        migrants = []
        while not self.migration_queue.empty():
            sender_id, receiver_id, solutions = self.migration_queue.get()
            if receiver_id == self.id and sender_id in self.neighbors:
                migrants.extend([sol for sol, _ in solutions])
        return migrants[:self.config.migration_size]

class AdaptiveParallelGA(ParallelGeneticAlgorithm):
    """Адаптивна паралельна версія генетичного алгоритму."""
    
    def __init__(self,
                 num_islands: int = 4,
                 generations_per_island: int = 50,
                 base_population_size: int = 50,
                 topology: MigrationTopology = MigrationTopology.RING):
        super().__init__(num_islands, generations_per_island, base_population_size)
        self.topology = topology
        self.name = "Adaptive Parallel GA"
    
    def create_topology(self) -> nx.Graph:
        """Створює граф зв'язків між островами."""
        G = nx.Graph()
        G.add_nodes_from(range(self.num_islands))
        
        if self.topology == MigrationTopology.RING:
            # Кільцева топологія
            for i in range(self.num_islands):
                G.add_edge(i, (i + 1) % self.num_islands)
        
        elif self.topology == MigrationTopology.FULLY_CONNECTED:
            # Повністю зв'язана топологія
            for i in range(self.num_islands):
                for j in range(i + 1, self.num_islands):
                    G.add_edge(i, j)
        
        elif self.topology == MigrationTopology.RANDOM:
            # Випадкова топологія (з гарантією зв'язності)
            while not nx.is_connected(G):
                G.add_edge(
                    random.randint(0, self.num_islands - 1),
                    random.randint(0, self.num_islands - 1)
                )
        
        elif self.topology == MigrationTopology.HIERARCHICAL:
            # Ієрархічна топологія
            levels = int(np.log2(self.num_islands))
            current_node = 0
            for level in range(levels):
                nodes_in_level = 2 ** level
                for i in range(nodes_in_level):
                    if current_node + 2 <= self.num_islands:
                        G.add_edge(current_node, current_node + 1)
                        G.add_edge(current_node, current_node + 2)
                    current_node += 1
        
        return G
    
    def optimize(self, initial_solution: Solution) -> Tuple[Solution, Dict]:
        """Виконує адаптивну паралельну оптимізацію."""
        start_time = time()
        
        # Створюємо топологію
        topology_graph = self.create_topology()
        
        # Створюємо спільну чергу для міграції
        with Manager() as manager:
            migration_queue = manager.Queue()
            
            # Створюємо адаптивні конфігурації для островів
            island_configs = []
            for i in range(self.num_islands):
                config = AdaptiveIslandConfig(
                    population_size=self.base_population_size + 
                                  random.randint(-10, 10),
                    mutation_rate=0.1 + random.uniform(-0.05, 0.05),
                    crossover_rate=0.8 + random.uniform(-0.1, 0.1),
                    local_search_freq=0.1 + random.uniform(-0.05, 0.05),
                    migration_freq=5 + random.randint(-2, 2),
                    migration_size=3 + random.randint(-1, 1)
                )
                island_configs.append(config)
            
            # Готуємо аргументи для кожного острова
            island_args = []
            for i, config in enumerate(island_configs):
                # Визначаємо сусідів для острова
                neighbors = set(nx.neighbors(topology_graph, i))
                
                args = (i, config, initial_solution, migration_queue, 
                       self.topology, neighbors)
                island_args.append(args)
            
            # Запускаємо острови паралельно
            with ProcessPoolExecutor(max_workers=self.num_islands) as executor:
                results = list(executor.map(self.run_adaptive_island, 
                                         island_args))
            
            # Знаходимо найкращий розв'язок серед усіх островів
            best_solution = min(
                [solution for _, solution in results],
                key=lambda x: x.compute_total_cost()
            )
        
        computation_time = time() - start_time
        metrics = OptimizationMetrics.calculate(
            best_solution,
            computation_time,
            self.generations_per_island * self.num_islands
        )
        
        return best_solution, asdict(metrics)
    
    @staticmethod
    def run_adaptive_island(args: Tuple) -> Tuple[int, Solution]:
        """Запускає адаптивний острів."""
        island_id, config, initial_solution, migration_queue, topology, neighbors = args
        
        island = AdaptiveIsland(island_id, config, initial_solution,
                              migration_queue, topology)
        island.neighbors = neighbors
        island.run_evolution(island.config.generations_per_island)
        
        return island_id, island.best_solution

def visualize_adaptive_parameters(island: AdaptiveIsland):
    """Візуалізує зміну параметрів острова."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    history = island.config.parameter_history
    generations = range(len(history['mutation_rate']))
    
    # Графік зміни mutation_rate
    ax1.plot(generations, history['mutation_rate'])
    ax1.set_title('Mutation Rate Adaptation')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Mutation Rate')
    ax1.grid(True)
    
    # Графік зміни crossover_rate
    ax2.plot(generations, history['crossover_rate'])
    ax2.set_title('Crossover Rate Adaptation')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Crossover Rate')
    ax2.grid(True)
    
    # Графік зміни local_search_freq
    ax3.plot(generations, history['local_search_freq'])
    ax3.set_title('Local Search Frequency Adaptation')
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Local Search Frequency')
    ax3.grid(True)
    
    # Графік різноманітності популяції
    ax4.plot(generations, history['diversity'])
    ax4.set_title('Population Diversity')
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Diversity')
    ax4.grid(True)
    
    plt.tight_layout()
    return fig

# Приклад використання:
if __name__ == "__main__":
    # Створюємо тестову задачу
    problem = generate_test_problem(
        num_warehouses=2,
        num_orders=30,
        num_trucks=5,
        area_size=100,
        seed=42
    )
    
    # Знаходимо початковий розв'язок
    initial_solution = greedy_solution(problem)
    print("\nПочатковий розв'язок:")
    print(f"Вартість: {initial_solution.compute_total_cost():.2f}")
    
    # Тестуємо різні топології
    topologies = [
        MigrationTopology.RING,
        MigrationTopology.FULLY_CONNECTED,
        MigrationTopology.RANDOM,
        MigrationTopology.HIERARCHICAL
    ]
    
    results = {}
    
    for topology in topologies:
        print(f"\nТестування топології: {topology.value}")
        
        adaptive_ga = AdaptiveParallelGA(
            num_islands=4,
            generations_per_island=50,
            base_population_size=50,
            topology=topology
        )
        
        solution, metrics = adaptive_ga.optimize(initial_solution)
        results[topology] = {
            'solution': solution,
            'metrics': metrics
        }
    
    # Порівнюємо результати
    print("\nПорівняння результатів для різних топологій:")
    print("-" * 60)
    print(f"{'Топологія':20} {'Вартість':15} {'Час (сек)':15}")
    print("-" * 60)
    
    for topology, result in results.items():
        cost = result['solution'].compute_total_cost()
        time = result['metrics']['computation_time']
        print(f"{topology.value:20} {cost:15.2f} {time:15.2f}")
    
    # Знаходимо найкращу топологію
    best_topology = min(results.items(), 
                       key=lambda x: x[1]['solution'].compute_total_cost())
    
    print("\nНайкраща топологія:", best_topology[0].value)
    best_solution = best_topology[1]['solution']
    
    # Візуалізуємо результати
    fig1 = visualize_solution(initial_solution, "Початковий розв'язок (жадібний)")
    fig2 = visualize_solution(best_solution, 
                            f"Найкращий розв'язок (Adaptive Parallel GA - {best_topology[0].value})")
    
    plt.show()
    
    print("\nЗагальні результати:")
    print(f"Початкова вартість: {initial_solution.compute_total_cost():.2f}")
    print(f"Кінцева вартість: {best_solution.compute_total_cost():.2f}")
    print(f"Покращення: {(initial_solution.compute_total_cost() - best_solution.compute_total_cost()):.2f} "
          f"({(initial_solution.compute_total_cost() - best_solution.compute_total_cost()) / initial_solution.compute_total_cost() * 100:.1f}%)")