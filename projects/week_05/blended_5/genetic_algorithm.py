from typing import List, Tuple, Dict, Optional
import random
from copy import deepcopy
import numpy as np
from logistics_base import LogisticsProblem, Order, Warehouse, Truck, Point
from greedy_solution import Solution, Route
from optimization_comparison import Optimizer

class ChromosomeEncoder:
    """Клас для кодування/декодування розв'язку в хромосому."""
    
    @staticmethod
    def encode(solution: Solution) -> List[int]:
        """
        Кодує розв'язок у хромосому.
        Формат: [склад_1, замовлення_1, ..., склад_2, замовлення_1, ...]
        Кожне замовлення кодується своїм ID.
        """
        chromosome = []
        for route in solution.routes:
            # Додаємо склад (зі зсувом для відрізнення від замовлень)
            chromosome.append(1000 + route.warehouse.id)
            # Додаємо замовлення
            for order in route.orders:
                chromosome.append(order.id)
        return chromosome
    
    @staticmethod
    def decode(chromosome: List[int], problem: LogisticsProblem) -> Solution:
        """Декодує хромосому в розв'язок."""
        solution = Solution(problem)
        current_route = None
        
        for gene in chromosome:
            if gene >= 1000:  # Це склад
                warehouse_id = gene - 1000
                warehouse = next(w for w in problem.warehouses if w.id == warehouse_id)
                # Шукаємо доступну вантажівку
                available_trucks = [t for t in problem.trucks 
                                 if not any(r.truck == t for r in solution.routes)]
                if available_trucks:
                    current_route = Route(
                        truck=available_trucks[0],
                        warehouse=warehouse
                    )
                    solution.add_route(current_route)
            else:  # Це замовлення
                if current_route is not None:
                    order = next(o for o in problem.orders if o.id == gene)
                    if order in solution.unassigned_orders:
                        if current_route.add_order(order):
                            solution.unassigned_orders.remove(order)
        
        return solution

class GeneticOperators:
    """Клас з генетичними операторами."""
    
    @staticmethod
    def crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Реалізує операцію кросовера (OX - Order Crossover).
        """
        if len(parent1) < 2 or len(parent2) < 2:
            return parent1[:], parent2[:]
            
        # Вибираємо дві точки розрізу
        point1, point2 = sorted(random.sample(range(len(parent1)), 2))
        
        # Створюємо нащадків
        child1 = [-1] * len(parent1)
        child2 = [-1] * len(parent2)
        
        # Копіюємо середні частини
        child1[point1:point2] = parent1[point1:point2]
        child2[point1:point2] = parent2[point1:point2]
        
        # Заповнюємо решту генів
        def fill_remaining(child: List[int], parent: List[int], other_parent: List[int]):
            used_genes = set(g for g in child if g != -1)
            genes_to_add = [g for g in parent + other_parent 
                          if g not in used_genes]
            idx = 0
            for i in range(len(child)):
                if child[i] == -1:
                    child[i] = genes_to_add[idx]
                    idx += 1
        
        fill_remaining(child1, parent1, parent2)
        fill_remaining(child2, parent2, parent1)
        
        return child1, child2
    
    @staticmethod
    def mutate(chromosome: List[int], mutation_rate: float = 0.1) -> List[int]:
        """
        Реалізує операцію мутації.
        """
        mutated = chromosome[:]
        
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Якщо це склад, залишаємо його на місці
                if mutated[i] >= 1000:
                    continue
                    
                # Для замовлення - обираємо випадкову позицію для обміну
                j = random.randint(0, len(mutated) - 1)
                if mutated[j] < 1000:  # Міняємо місцями тільки замовлення
                    mutated[i], mutated[j] = mutated[j], mutated[i]
        
        return mutated

class GeneticAlgorithm(Optimizer):
    """Реалізація генетичного алгоритму."""
    
    def __init__(self,
                 population_size: int = 50,
                 num_generations: int = 100,
                 mutation_rate: float = 0.1,
                 elite_size: int = 5):
        """
        Args:
            population_size: розмір популяції
            num_generations: кількість поколінь
            mutation_rate: ймовірність мутації
            elite_size: кількість найкращих особин, що переходять у наступне покоління
        """
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.encoder = ChromosomeEncoder()
        self.operators = GeneticOperators()
    
    @property
    def name(self) -> str:
        return "Genetic Algorithm"
    
    def _create_initial_population(self, initial_solution: Solution) -> List[List[int]]:
        """Створює початкову популяцію на основі жадібного розв'язку."""
        population = [self.encoder.encode(initial_solution)]
        
        while len(population) < self.population_size:
            # Створюємо варіації початкового розв'язку
            chromosome = population[0][:]
            # Виконуємо випадкові перестановки
            for _ in range(random.randint(1, 5)):
                chromosome = self.operators.mutate(chromosome, self.mutation_rate)
            population.append(chromosome)
        
        return population
    
    def _evaluate_population(self, population: List[List[int]], 
                           problem: LogisticsProblem) -> List[Tuple[List[int], float]]:
        """Оцінює популяцію та сортує за якістю."""
        evaluated = []
        for chromosome in population:
            solution = self.encoder.decode(chromosome, problem)
            fitness = -solution.compute_total_cost()  # негативне, бо мінімізуємо
            evaluated.append((chromosome, fitness))
        
        return sorted(evaluated, key=lambda x: x[1], reverse=True)
    
    def _select_parents(self, evaluated_population: List[Tuple[List[int], float]], 
                       num_parents: int) -> List[List[int]]:
        """Вибирає батьків за допомогою турнірної селекції."""
        parents = []
        for _ in range(num_parents):
            # Вибираємо випадкових кандидатів для турніру
            tournament = random.sample(evaluated_population, 
                                    k=min(3, len(evaluated_population)))
            # Вибираємо найкращого з турніру
            winner = max(tournament, key=lambda x: x[1])
            parents.append(winner[0])
        return parents
    
    def optimize(self, initial_solution: Solution) -> Tuple[Solution, Dict]:
        """Виконує оптимізацію генетичним алгоритмом."""
        start_time = time()
        problem = initial_solution.problem
        
        # Створюємо початкову популяцію
        population = self._create_initial_population(initial_solution)
        best_solution = initial_solution
        best_fitness = -initial_solution.compute_total_cost()
        
        generation = 0
        generations_without_improvement = 0
        
        while generation < self.num_generations:
            # Оцінюємо популяцію
            evaluated = self._evaluate_population(population, problem)
            
            # Перевіряємо на покращення
            current_best_fitness = evaluated[0][1]
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_solution = self.encoder.decode(evaluated[0][0], problem)
                generations_without_improvement = 0
                print(f"Generation {generation}: Found better solution. "
                      f"Cost: {-best_fitness:.2f}")
            else:
                generations_without_improvement += 1
            
            # Відбираємо еліту
            new_population = [chr for chr, _ in evaluated[:self.elite_size]]
            
            # Створюємо нове покоління
            while len(new_population) < self.population_size:
                # Вибираємо батьків
                parents = self._select_parents(evaluated, 2)
                
                # Схрещування
                child1, child2 = self.operators.crossover(parents[0], parents[1])
                
                # Мутація
                child1 = self.operators.mutate(child1, self.mutation_rate)
                child2 = self.operators.mutate(child2, self.mutation_rate)
                
                new_population.extend([child1, child2])
            
            # Обмежуємо розмір популяції
            population = new_population[:self.population_size]
            generation += 1
        
        computation_time = time() - start_time
        
        metrics = OptimizationMetrics.calculate(best_solution, computation_time,
                                              generation)
        
        print(f"\nGenetic Algorithm finished after {generation} generations")
        print(f"Initial cost: {initial_solution.compute_total_cost():.2f}")
        print(f"Final cost: {-best_fitness:.2f}")
        print(f"Improvement: {(initial_solution.compute_total_cost() + best_fitness):.2f} "
              f"({(initial_solution.compute_total_cost() + best_fitness) / initial_solution.compute_total_cost() * 100:.1f}%)")
        
        return best_solution, asdict(metrics)

# Приклад використання:
if __name__ == "__main__":
    # Створюємо тестову задачу
    problem = generate_test_problem(
        num_warehouses=2,
        num_orders=20,
        num_trucks=4,
        area_size=100,
        seed=42
    )
    
    # Знаходимо початковий розв'язок жадібним алгоритмом
    initial_solution = greedy_solution(problem)
    print("\nПочатковий розв'язок:")
    print(f"Вартість: {initial_solution.compute_total_cost():.2f}")
    
    # Налаштовуємо та запускаємо генетичний алгоритм
    ga = GeneticAlgorithm(
        population_size=50,
        num_generations=100,
        mutation_rate=0.1,
        elite_size=5
    )
    
    improved_solution, metrics = ga.optimize(initial_solution)
    
    # Оновлюємо список оптимізаторів для порівняння
    optimizers = [
        HillClimbingOptimizer(max_iterations=1000),
        SimulatedAnnealingOptimizer(max_iterations=5000),
        AdaptiveSAOptimizer(max_iterations=5000),
        GeneticAlgorithm(num_generations=100)
    ]
    
    # Запускаємо порівняння
    comparison = OptimizationComparison(problem, optimizers)
    results_df = comparison.run_comparison(num_runs=3)
    
    # Візуалізуємо результати
    fig1 = visualize_solution(initial_solution, "Початковий розв'язок (жадібний)")
    fig2 = visualize_solution(improved_solution, "Покращений розв'язок (GA)")
    fig3 = visualize_comparison(results_df)
    plt.show()
