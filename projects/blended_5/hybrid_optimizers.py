from typing import List, Dict, Tuple, Optional, Callable
from abc import ABC, abstractmethod
import random
from copy import deepcopy
import numpy as np
from logistics_base import LogisticsProblem
from greedy_solution import Solution
from optimization_comparison import Optimizer, OptimizationMetrics

class HybridOptimizer(Optimizer):
    """Базовий клас для гібридних оптимізаторів."""
    
    def __init__(self, name: str):
        self.name = name
        self._optimization_history = []
    
    @property
    def algorithm_name(self) -> str:
        return self.name
    
    def record_progress(self, iteration: int, solution: Solution,
                       method: str, cost: float):
        """Записує прогрес оптимізації."""
        self._optimization_history.append({
            'iteration': iteration,
            'method': method,
            'cost': cost
        })

class MemoryPool:
    """Клас для зберігання та управління найкращими розв'язками."""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.solutions: List[Tuple[Solution, float]] = []
    
    def add(self, solution: Solution):
        """Додає новий розв'язок до пулу."""
        cost = solution.compute_total_cost()
        
        # Перевіряємо, чи розв'язок вже є в пулі
        if not any(abs(existing_cost - cost) < 1e-6 
                  for _, existing_cost in self.solutions):
            self.solutions.append((deepcopy(solution), cost))
            # Сортуємо за вартістю (зростання)
            self.solutions.sort(key=lambda x: x[1])
            # Обмежуємо розмір пулу
            if len(self.solutions) > self.max_size:
                self.solutions.pop()
    
    def get_random(self) -> Optional[Solution]:
        """Повертає випадковий розв'язок з пулу."""
        if not self.solutions:
            return None
        return deepcopy(random.choice(self.solutions)[0])
    
    def get_best(self) -> Optional[Solution]:
        """Повертає найкращий розв'язок."""
        if not self.solutions:
            return None
        return deepcopy(self.solutions[0][0])

class GAWithLocalSearch(HybridOptimizer):
    """Гібрид генетичного алгоритму з локальним пошуком."""
    
    def __init__(self,
                 population_size: int = 50,
                 num_generations: int = 100,
                 local_search_freq: float = 0.1,
                 memory_pool_size: int = 10):
        """
        Args:
            population_size: розмір популяції
            num_generations: кількість поколінь
            local_search_freq: частота застосування локального пошуку
            memory_pool_size: розмір пулу пам'яті
        """
        super().__init__("GA with Local Search")
        self.population_size = population_size
        self.num_generations = num_generations
        self.local_search_freq = local_search_freq
        self.memory_pool = MemoryPool(memory_pool_size)
        self.ga = GeneticAlgorithm(population_size, num_generations)
        self.local_search = HillClimbingOptimizer(max_iterations=100)
    
    def optimize(self, initial_solution: Solution) -> Tuple[Solution, Dict]:
        """Виконує гібридну оптимізацію."""
        start_time = time()
        
        # Ініціалізуємо популяцію
        population = self.ga._create_initial_population(initial_solution)
        best_solution = initial_solution
        best_fitness = -initial_solution.compute_total_cost()
        
        generation = 0
        total_iterations = 0
        
        while generation < self.num_generations:
            # Генеруємо звіт
    framework.generate_report()
    
    # Окремо аналізуємо роботу адаптивного гібридного оптимізатора
    adaptive_hybrid = AdaptiveHybridOptimizer()
    solution, metrics = adaptive_hybrid.optimize(initial_solution)
    
    # Візуалізуємо результати
    fig1 = visualize_solution(initial_solution, "Початковий розв'язок (жадібний)")
    fig2 = visualize_solution(solution, "Покращений розв'язок (Adaptive Hybrid)")
    fig3 = visualize_hybrid_performance(metrics)
    plt.show()
    
    print("\nРезультати адаптивного гібридного алгоритму:")
    print(f"Початкова вартість: {initial_solution.compute_total_cost():.2f}")
    print(f"Кінцева вартість: {solution.compute_total_cost():.2f}")
    print(f"Покращення: {(initial_solution.compute_total_cost() - solution.compute_total_cost()):.2f} "
          f"({(initial_solution.compute_total_cost() - solution.compute_total_cost()) / initial_solution.compute_total_cost() * 100:.1f}%)")
    
    print("\nПродуктивність методів:")
    for method, improvement in metrics['method_performance'].items():
        print(f"{method}: {improvement:.2f}")
            evaluated = self.ga._evaluate_population(population,
                                                  initial_solution.problem)
            
            current_best_chromosome = evaluated[0][0]
            current_solution = self.ga.encoder.decode(
                current_best_chromosome,
                initial_solution.problem
            )
            current_fitness = -current_solution.compute_total_cost()
            
            # Записуємо прогрес
            self.record_progress(total_iterations, current_solution,
                               "GA", -current_fitness)
            
            # Перевіряємо на покращення
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_solution = deepcopy(current_solution)
                # Додаємо до пулу пам'яті
                self.memory_pool.add(best_solution)
                print(f"Generation {generation}: Found better solution. "
                      f"Cost: {-best_fitness:.2f}")
            
            # Локальний пошук
            if random.random() < self.local_search_freq:
                # Вибираємо випадковий розв'язок з кращої половини популяції
                candidate_idx = random.randint(0, len(evaluated) // 2)
                candidate = self.ga.encoder.decode(
                    evaluated[candidate_idx][0],
                    initial_solution.problem
                )
                
                # Застосовуємо локальний пошук
                improved_solution, _ = self.local_search.optimize(candidate)
                improved_fitness = -improved_solution.compute_total_cost()
                
                self.record_progress(total_iterations, improved_solution,
                                  "Local Search", -improved_fitness)
                
                if improved_fitness > best_fitness:
                    best_fitness = improved_fitness
                    best_solution = deepcopy(improved_solution)
                    self.memory_pool.add(best_solution)
                    print(f"Local Search improved solution. "
                          f"Cost: {-best_fitness:.2f}")
                
                # Додаємо покращений розв'язок назад до популяції
                improved_chromosome = self.ga.encoder.encode(improved_solution)
                population = population[:-1] + [improved_chromosome]
            
            # Створюємо нове покоління
            new_population = [chr for chr, _ in evaluated[:self.ga.elite_size]]
            
            # Додаємо розв'язок з пулу пам'яті
            if random.random() < 0.1 and self.memory_pool.solutions:
                memory_solution = self.memory_pool.get_random()
                if memory_solution:
                    memory_chromosome = self.ga.encoder.encode(memory_solution)
                    new_population.append(memory_chromosome)
            
            # Генеруємо решту популяції
            while len(new_population) < self.population_size:
                parents = self.ga._select_parents(evaluated, 2)
                child1, child2 = self.ga.operators.crossover(parents[0], parents[1])
                child1 = self.ga.operators.mutate(child1, self.ga.mutation_rate)
                child2 = self.ga.operators.mutate(child2, self.ga.mutation_rate)
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
            generation += 1
            total_iterations += 1
        
        computation_time = time() - start_time
        metrics = OptimizationMetrics.calculate(best_solution, computation_time,
                                              total_iterations)
        
        return best_solution, asdict(metrics)

class AdaptiveHybridOptimizer(HybridOptimizer):
    """Адаптивний гібридний оптимізатор."""
    
    def __init__(self,
                 max_iterations: int = 5000,
                 memory_pool_size: int = 10):
        """
        Args:
            max_iterations: максимальна кількість ітерацій
            memory_pool_size: розмір пулу пам'яті
        """
        super().__init__("Adaptive Hybrid")
        self.max_iterations = max_iterations
        self.memory_pool = MemoryPool(memory_pool_size)
        
        # Створюємо оптимізатори
        self.ga = GeneticAlgorithm(population_size=30, num_generations=50)
        self.sa = AdaptiveSimulatedAnnealing(max_iterations=1000)
        self.local_search = HillClimbingOptimizer(max_iterations=100)
    
    def optimize(self, initial_solution: Solution) -> Tuple[Solution, Dict]:
        """Виконує адаптивну гібридну оптимізацію."""
        start_time = time()
        
        best_solution = initial_solution
        best_cost = initial_solution.compute_total_cost()
        
        iteration = 0
        stagnation_counter = 0
        method_performance = {
            'GA': 0,
            'SA': 0,
            'Local Search': 0
        }
        
        while iteration < self.max_iterations:
            # Вибираємо метод на основі їх продуктивності
            if iteration < 100:
                # На початку пробуємо всі методи рівномірно
                method = random.choice(['GA', 'SA', 'Local Search'])
            else:
                # Вибираємо метод зважено на основі їх успішності
                total_improvements = sum(method_performance.values()) + 1e-6
                probabilities = {
                    method: (score + 1e-6) / total_improvements 
                    for method, score in method_performance.items()
                }
                method = random.choices(
                    list(probabilities.keys()),
                    weights=list(probabilities.values())
                )[0]
            
            # Застосовуємо вибраний метод
            if method == 'GA':
                current_solution, _ = self.ga.optimize(best_solution)
            elif method == 'SA':
                current_solution, _ = self.sa.optimize(best_solution)
            else:  # Local Search
                current_solution, _ = self.local_search.optimize(best_solution)
            
            current_cost = current_solution.compute_total_cost()
            
            # Записуємо прогрес
            self.record_progress(iteration, current_solution, method, current_cost)
            
            # Оновлюємо найкращий розв'язок
            if current_cost < best_cost:
                improvement = best_cost - current_cost
                method_performance[method] += improvement
                best_cost = current_cost
                best_solution = deepcopy(current_solution)
                self.memory_pool.add(best_solution)
                stagnation_counter = 0
                print(f"Iteration {iteration}: {method} improved solution. "
                      f"Cost: {best_cost:.2f}")
            else:
                stagnation_counter += 1
            
            # Диверсифікація при стагнації
            if stagnation_counter >= 100:
                # Спробуємо взяти розв'язок з пулу пам'яті
                memory_solution = self.memory_pool.get_random()
                if memory_solution:
                    best_solution = memory_solution
                    best_cost = best_solution.compute_total_cost()
                    print(f"Iteration {iteration}: Diversifying from memory pool. "
                          f"Cost: {best_cost:.2f}")
                stagnation_counter = 0
            
            iteration += 1
        
        computation_time = time() - start_time
        metrics = OptimizationMetrics.calculate(best_solution, computation_time,
                                              iteration)
        
        # Додаємо інформацію про використання методів
        metrics.update({
            'method_performance': method_performance,
            'optimization_history': self._optimization_history
        })
        
        return best_solution, metrics

def visualize_hybrid_performance(metrics: Dict):
    """Візуалізує роботу гібридного алгоритму."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Графік прогресу оптимізації
    history = pd.DataFrame(metrics['optimization_history'])
    for method in history['method'].unique():
        method_data = history[history['method'] == method]
        ax1.scatter(method_data['iteration'], method_data['cost'],
                   label=method, alpha=0.6)
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Solution Cost')
    ax1.set_title('Optimization Progress')
    ax1.legend()
    ax1.grid(True)
    
    # Графік продуктивності методів
    method_performance = metrics['method_performance']
    ax2.bar(method_performance.keys(), method_performance.values())
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Total Improvement')
    ax2.set_title('Method Performance')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

if name == "__main__":
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

    # Створюємо та запускаємо гібридні оптимізатори
    optimizers = [
        GAWithLocalSearch(),
        AdaptiveHybridOptimizer(),
        GeneticAlgorithm(num_generations=100),
        AdaptiveSAOptimizer(max_iterations=5000)
    ]

    # Запускаємо порівняння
    framework = TestingFramework(optimizers)
    framework.run_tests(num_runs=3)

    # Генеруємо звіт
    framework.generate_report()

    # Окремо аналізуємо роботу адаптивного гібридного оптимізатора
    adaptive_hybrid = AdaptiveHybridOptimizer()
    solution, metrics = adaptive_hybrid.optimize(initial_solution)

    # Візуалізуємо результати
    fig1 = visualize_solution(initial_solution, "Початковий розв'язок (жадібний)")
    fig2 = visualize_solution(solution, "Покращений розв'язок (Adaptive Hybrid)")
    fig3 = visualize_hybrid_performance(metrics)
    plt.show()

    print("\nРезультати адаптивного гібридного алгоритму:")
    print(f"Початкова вартість: {initial_solution.compute_total_cost():.2f}")
    print(f"Кінцева вартість: {solution.compute_total_cost():.2f}")
    print(f"Покращення: {(initial_solution.compute_total_cost() - solution.compute_total_cost()):.2f} "
        f"({(initial_solution.compute_total_cost() - solution.compute_total_cost()) / initial_solution.compute_total_cost() * 100:.1f}%)")

    print("\nПродуктивність методів:")
    for method, improvement in metrics['method_performance'].items():
        print(f"{method}: {improvement:.2f}")