from multiprocessing import Process, Queue, Pool, Manager
from typing import List, Dict, Tuple, Optional
import numpy as np
from copy import deepcopy
import time
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

@dataclass
class IslandConfig:
    """Конфігурація для острова в паралельному GA."""
    population_size: int
    mutation_rate: float
    crossover_rate: float
    local_search_freq: float
    migration_freq: int  # частота міграції (в поколіннях)
    migration_size: int  # кількість особин для міграції

class Island:
    """Представляє окремий острів в паралельному GA."""
    
    def __init__(self, 
                 island_id: int,
                 config: IslandConfig,
                 initial_solution: Solution,
                 migration_queue: Queue):
        """
        Args:
            island_id: ідентифікатор острова
            config: конфігурація острова
            initial_solution: початковий розв'язок
            migration_queue: черга для міграції
        """
        self.id = island_id
        self.config = config
        self.initial_solution = initial_solution
        self.migration_queue = migration_queue
        
        self.population = []
        self.best_solution = None
        self.generation = 0
        
        # Створюємо оптимізатори
        self.ga = GeneticAlgorithm(
            population_size=config.population_size,
            mutation_rate=config.mutation_rate
        )
        self.local_search = HillClimbingOptimizer(max_iterations=50)
    
    def initialize_population(self):
        """Ініціалізує популяцію острова."""
        self.population = self.ga._create_initial_population(self.initial_solution)
        self.evaluate_population()
    
    def evaluate_population(self) -> List[Tuple[List[int], float]]:
        """Оцінює популяцію та повертає відсортований список (хромосома, якість)."""
        return self.ga._evaluate_population(self.population, 
                                         self.initial_solution.problem)
    
    def migrate_solutions(self, evaluated_population: List[Tuple[List[int], float]]):
        """Відправляє найкращі розв'язки в чергу міграції."""
        if self.generation % self.config.migration_freq == 0:
            best_solutions = evaluated_population[:self.config.migration_size]
            self.migration_queue.put((self.id, best_solutions))
    
    def receive_migrants(self) -> List[List[int]]:
        """Отримує мігрантів з інших островів."""
        migrants = []
        while not self.migration_queue.empty():
            sender_id, solutions = self.migration_queue.get()
            if sender_id != self.id:  # не беремо власні розв'язки
                migrants.extend([sol for sol, _ in solutions])
        return migrants[:self.config.migration_size]  # обмежуємо кількість
    
    def run_evolution(self, num_generations: int):
        """Запускає еволюційний процес на острові."""
        self.initialize_population()
        
        for generation in range(num_generations):
            self.generation = generation
            
            # Оцінюємо популяцію
            evaluated = self.evaluate_population()
            
            # Міграція
            self.migrate_solutions(evaluated)
            migrants = self.receive_migrants()
            
            # Створюємо нове покоління
            new_population = [chr for chr, _ in evaluated[:self.ga.elite_size]]
            
            # Додаємо мігрантів
            if migrants:
                new_population.extend(migrants)
            
            # Локальний пошук
            if random.random() < self.config.local_search_freq:
                # Вибираємо випадковий розв'язок з кращої половини
                candidate_idx = random.randint(0, len(evaluated) // 2)
                candidate = self.ga.encoder.decode(
                    evaluated[candidate_idx][0],
                    self.initial_solution.problem
                )
                
                # Застосовуємо локальний пошук
                improved_solution, _ = self.local_search.optimize(candidate)
                improved_chromosome = self.ga.encoder.encode(improved_solution)
                new_population.append(improved_chromosome)
            
            # Генеруємо решту популяції
            while len(new_population) < self.config.population_size:
                parents = self.ga._select_parents(evaluated, 2)
                child1, child2 = self.ga.operators.crossover(parents[0], parents[1])
                
                if random.random() < self.config.crossover_rate:
                    child1 = self.ga.operators.mutate(child1, 
                                                    self.config.mutation_rate)
                if random.random() < self.config.crossover_rate:
                    child2 = self.ga.operators.mutate(child2, 
                                                    self.config.mutation_rate)
                
                new_population.extend([child1, child2])
            
            self.population = new_population[:self.config.population_size]
            
            # Оновлюємо найкращий розв'язок
            current_best = self.ga.encoder.decode(
                evaluated[0][0],
                self.initial_solution.problem
            )
            
            if (self.best_solution is None or 
                current_best.compute_total_cost() < 
                self.best_solution.compute_total_cost()):
                self.best_solution = deepcopy(current_best)
                print(f"Island {self.id}, Generation {generation}: "
                      f"New best solution with cost "
                      f"{self.best_solution.compute_total_cost():.2f}")

class ParallelGeneticAlgorithm(HybridOptimizer):
    """Паралельна версія генетичного алгоритму з островною моделлю."""
    
    def __init__(self,
                 num_islands: int = 4,
                 generations_per_island: int = 50,
                 base_population_size: int = 50):
        """
        Args:
            num_islands: кількість островів
            generations_per_island: кількість поколінь на кожному острові
            base_population_size: базовий розмір популяції
        """
        super().__init__("Parallel GA")
        self.num_islands = num_islands
        self.generations_per_island = generations_per_island
        
        # Створюємо різні конфігурації для островів
        self.island_configs = []
        for i in range(num_islands):
            # Варіюємо параметри для різних островів
            config = IslandConfig(
                population_size=base_population_size + random.randint(-10, 10),
                mutation_rate=0.1 + random.uniform(-0.05, 0.05),
                crossover_rate=0.8 + random.uniform(-0.1, 0.1),
                local_search_freq=0.1 + random.uniform(-0.05, 0.05),
                migration_freq=5 + random.randint(-2, 2),
                migration_size=3 + random.randint(-1, 1)
            )
            self.island_configs.append(config)
    
    def run_island(self, args: Tuple[int, IslandConfig, Solution, Queue]
                  ) -> Tuple[int, Solution]:
        """Запускає еволюцію на одному острові."""
        island_id, config, initial_solution, migration_queue = args
        
        island = Island(island_id, config, initial_solution, migration_queue)
        island.run_evolution(self.generations_per_island)
        
        return island_id, island.best_solution
    
    def optimize(self, initial_solution: Solution) -> Tuple[Solution, Dict]:
        """Виконує паралельну оптимізацію."""
        start_time = time()
        
        # Створюємо спільну чергу для міграції
        with Manager() as manager:
            migration_queue = manager.Queue()
            
            # Готуємо аргументи для кожного острова
            island_args = [
                (i, config, initial_solution, migration_queue)
                for i, config in enumerate(self.island_configs)
            ]
            
            # Запускаємо острови паралельно
            with ProcessPoolExecutor(max_workers=self.num_islands) as executor:
                results = list(executor.map(self.run_island, island_args))
            
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
    
    # Створюємо та запускаємо паралельний GA
    parallel_ga = ParallelGeneticAlgorithm(
        num_islands=4,
        generations_per_island=50,
        base_population_size=50
    )
    
    solution, metrics = parallel_ga.optimize(initial_solution)
    
    # Візуалізуємо результати
    fig1 = visualize_solution(initial_solution, "Початковий розв'язок (жадібний)")
    fig2 = visualize_solution(solution, "Покращений розв'язок (Parallel GA)")
    plt.show()
    
    print("\nРезультати паралельного GA:")
    print(f"Початкова вартість: {initial_solution.compute_total_cost():.2f}")
    print(f"Кінцева вартість: {solution.compute_total_cost():.2f}")
    print(f"Покращення: {(initial_solution.compute_total_cost() - solution.compute_total_cost()):.2f} "
          f"({(initial_solution.compute_total_cost() - solution.compute_total_cost()) / initial_solution.compute_total_cost() * 100:.1f}%)")
    print(f"Час обчислень: {metrics['computation_time']:.2f} сек")
