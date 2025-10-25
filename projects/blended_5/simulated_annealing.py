import math
from dataclasses import dataclass
from typing import List, Tuple, Dict
import random
from copy import deepcopy
import numpy as np
from logistics_base import LogisticsProblem
from greedy_solution import Solution
from local_search import generate_neighbor

@dataclass
class SimulatedAnnealingStats:
    """Клас для збору статистики роботи алгоритму."""
    iterations: int = 0
    accepted_moves: int = 0
    improving_moves: int = 0
    deteriorating_moves: int = 0
    temperature_history: List[float] = None
    cost_history: List[float] = None
    best_cost_history: List[float] = None
    
    def acceptance_rate(self) -> float:
        """Повертає відсоток прийнятих ходів."""
        return self.accepted_moves / self.iterations if self.iterations > 0 else 0
    
    def improvement_rate(self) -> float:
        """Повертає відсоток покращуючих ходів."""
        return self.improving_moves / self.iterations if self.iterations > 0 else 0

class SimulatedAnnealing:
    """Реалізація алгоритму імітації відпалу."""
    
    def __init__(self,
                 initial_temp: float = 100.0,
                 final_temp: float = 1.0,
                 alpha: float = 0.98,
                 iterations_per_temp: int = 100,
                 max_iterations: int = 10000):
        """
        Args:
            initial_temp: початкова температура
            final_temp: кінцева температура
            alpha: коефіцієнт охолодження (0 < alpha < 1)
            iterations_per_temp: кількість ітерацій на кожній температурі
            max_iterations: максимальна кількість ітерацій
        """
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.alpha = alpha
        self.iterations_per_temp = iterations_per_temp
        self.max_iterations = max_iterations
        self.stats = None
    
    def _acceptance_probability(self, 
                              current_cost: float,
                              new_cost: float,
                              temperature: float) -> float:
        """Обчислює ймовірність прийняття нового розв'язку."""
        if new_cost < current_cost:
            return 1.0
        return math.exp(-(new_cost - current_cost) / temperature)
    
    def _initialize_stats(self):
        """Ініціалізує збір статистики."""
        self.stats = SimulatedAnnealingStats()
        self.stats.temperature_history = []
        self.stats.cost_history = []
        self.stats.best_cost_history = []
    
    def optimize(self, initial_solution: Solution) -> Tuple[Solution, SimulatedAnnealingStats]:
        """
        Виконує оптимізацію розв'язку методом імітації відпалу.
        
        Args:
            initial_solution: початковий розв'язок
            
        Returns:
            Tuple[Solution, SimulatedAnnealingStats]: кращий знайдений розв'язок та статистика
        """
        self._initialize_stats()
        
        current_solution = deepcopy(initial_solution)
        best_solution = deepcopy(current_solution)
        
        current_cost = current_solution.compute_total_cost()
        best_cost = current_cost
        
        temperature = self.initial_temp
        iteration = 0
        
        while (iteration < self.max_iterations and 
               temperature > self.final_temp):
            
            for _ in range(self.iterations_per_temp):
                # Генеруємо сусідній розв'язок
                new_solution, operation = generate_neighbor(current_solution)
                new_cost = new_solution.compute_total_cost()
                
                # Обчислюємо ймовірність прийняття
                acceptance_prob = self._acceptance_probability(
                    current_cost, new_cost, temperature
                )
                
                self.stats.iterations += 1
                
                # Вирішуємо, чи приймати новий розв'язок
                if random.random() < acceptance_prob:
                    current_solution = new_solution
                    current_cost = new_cost
                    self.stats.accepted_moves += 1
                    
                    if new_cost < current_cost:
                        self.stats.improving_moves += 1
                    else:
                        self.stats.deteriorating_moves += 1
                    
                    # Оновлюємо найкращий розв'язок
                    if new_cost < best_cost:
                        best_solution = deepcopy(new_solution)
                        best_cost = new_cost
                        print(f"Iteration {iteration}: Found better solution with {operation} "
                              f"operation. New cost: {best_cost:.2f}")
                
                # Зберігаємо статистику
                self.stats.temperature_history.append(temperature)
                self.stats.cost_history.append(current_cost)
                self.stats.best_cost_history.append(best_cost)
                
                iteration += 1
                if iteration >= self.max_iterations:
                    break
            
            # Зменшуємо температуру
            temperature *= self.alpha
        
        print(f"\nSimulated Annealing finished after {iteration} iterations")
        print(f"Initial cost: {initial_solution.compute_total_cost():.2f}")
        print(f"Final cost: {best_cost:.2f}")
        print(f"Improvement: {(initial_solution.compute_total_cost() - best_cost):.2f} "
              f"({(initial_solution.compute_total_cost() - best_cost) / initial_solution.compute_total_cost() * 100:.1f}%)")
        print(f"Acceptance rate: {self.stats.acceptance_rate():.1%}")
        print(f"Improvement rate: {self.stats.improvement_rate():.1%}")
        
        return best_solution, self.stats

def visualize_sa_stats(stats: SimulatedAnnealingStats):
    """Візуалізує статистику роботи алгоритму імітації відпалу."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Графік температури
    ax1.plot(stats.temperature_history, label='Temperature')
    ax1.set_yscale('log')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Temperature')
    ax1.set_title('Temperature Cooling Schedule')
    ax1.grid(True)
    ax1.legend()
    
    # Графік вартості розв'язку
    ax2.plot(stats.cost_history, label='Current Cost', alpha=0.5)
    ax2.plot(stats.best_cost_history, label='Best Cost', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Solution Cost')
    ax2.set_title('Solution Cost over Iterations')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    return fig

# Приклад використання:
if __name__ == "__main__":
    # Створюємо тестову задачу
    problem = generate_test_problem(
        num_warehouses=2,
        num_orders=15,
        num_trucks=4,
        area_size=100,
        seed=42
    )
    
    # Знаходимо початковий розв'язок жадібним алгоритмом
    initial_solution = greedy_solution(problem)
    print("\nПочатковий розв'язок:")
    print(f"Вартість: {initial_solution.compute_total_cost():.2f}")
    
    # Налаштовуємо та запускаємо імітацію відпалу
    sa = SimulatedAnnealing(
        initial_temp=100.0,
        final_temp=1.0,
        alpha=0.98,
        iterations_per_temp=50,
        max_iterations=5000
    )
    
    improved_solution, stats = sa.optimize(initial_solution)
    
    # Візуалізуємо результати
    fig1 = visualize_solution(initial_solution, "Початковий розв'язок (жадібний)")
    fig2 = visualize_solution(improved_solution, "Покращений розв'язок (SA)")
    fig3 = visualize_sa_stats(stats)
    plt.show()
