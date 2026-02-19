from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
from copy import deepcopy
import math
from logistics_base import LogisticsProblem
from greedy_solution import Solution
from local_search import generate_neighbor

@dataclass
class AdaptiveParameters:
    """Параметри для адаптивного налаштування."""
    temperature: float
    acceptance_rate: float = 0.0
    improvement_rate: float = 0.0
    last_improvements: List[float] = field(default_factory=list)
    stagnation_counter: int = 0
    
    # Цільові значення для адаптації
    target_acceptance_rate: float = 0.3  # бажаний рівень прийняття нових розв'язків
    min_temperature: float = 0.1
    max_temperature: float = 1000.0
    
    # Параметри для виявлення стагнації
    stagnation_window: int = 100  # вікно для аналізу стагнації
    stagnation_threshold: float = 0.001  # поріг відносного покращення
    
    def update_rates(self, accepted: bool, improved: bool):
        """Оновлює статистику прийняття та покращення."""
        # Експоненційне згладжування для rates
        alpha = 0.02  # коефіцієнт згладжування
        self.acceptance_rate = (1 - alpha) * self.acceptance_rate + alpha * float(accepted)
        self.improvement_rate = (1 - alpha) * self.improvement_rate + alpha * float(improved)
    
    def adapt_temperature(self):
        """Адаптивно налаштовує температуру."""
        if self.acceptance_rate < self.target_acceptance_rate - 0.1:
            # Збільшуємо температуру, якщо приймається замало розв'язків
            self.temperature *= 1.1
        elif self.acceptance_rate > self.target_acceptance_rate + 0.1:
            # Зменшуємо температуру, якщо приймається забагато розв'язків
            self.temperature *= 0.9
            
        # Обмежуємо температуру
        self.temperature = np.clip(self.temperature, 
                                 self.min_temperature,
                                 self.max_temperature)
    
    def check_stagnation(self, current_cost: float) -> bool:
        """Перевіряє наявність стагнації."""
        self.last_improvements.append(current_cost)
        if len(self.last_improvements) > self.stagnation_window:
            self.last_improvements.pop(0)
            
            # Обчислюємо відносне покращення
            if len(self.last_improvements) >= 2:
                relative_improvement = abs(
                    (self.last_improvements[-1] - self.last_improvements[0]) /
                    self.last_improvements[0]
                )
                
                if relative_improvement < self.stagnation_threshold:
                    self.stagnation_counter += 1
                    return True
        
        self.stagnation_counter = 0
        return False
    
    def handle_stagnation(self):
        """Реагує на стагнацію."""
        # Збільшуємо температуру для диверсифікації пошуку
        self.temperature *= 2.0
        self.temperature = min(self.temperature, self.max_temperature)
        
        # Скидаємо лічильники
        self.last_improvements.clear()
        self.stagnation_counter = 0

class AdaptiveSimulatedAnnealing:
    """Реалізація адаптивного алгоритму імітації відпалу."""
    
    def __init__(self,
                 initial_temp: float = 100.0,
                 max_iterations: int = 10000,
                 max_stagnation_iterations: int = 1000):
        """
        Args:
            initial_temp: початкова температура
            max_iterations: максимальна кількість ітерацій
            max_stagnation_iterations: максимальна кількість ітерацій стагнації
        """
        self.params = AdaptiveParameters(temperature=initial_temp)
        self.max_iterations = max_iterations
        self.max_stagnation_iterations = max_stagnation_iterations
        self.stats = None
    
    def _initialize_stats(self):
        """Ініціалізує статистику."""
        self.stats = {
            'iterations': [],
            'temperature': [],
            'current_cost': [],
            'best_cost': [],
            'acceptance_rate': [],
            'improvement_rate': [],
            'stagnation_events': []
        }
    
    def _update_stats(self, iteration: int, current_cost: float, 
                     best_cost: float, stagnation: bool):
        """Оновлює статистику."""
        self.stats['iterations'].append(iteration)
        self.stats['temperature'].append(self.params.temperature)
        self.stats['current_cost'].append(current_cost)
        self.stats['best_cost'].append(best_cost)
        self.stats['acceptance_rate'].append(self.params.acceptance_rate)
        self.stats['improvement_rate'].append(self.params.improvement_rate)
        self.stats['stagnation_events'].append(stagnation)
    
    def _acceptance_probability(self, current_cost: float,
                              new_cost: float) -> float:
        """Обчислює ймовірність прийняття нового розв'язку."""
        if new_cost < current_cost:
            return 1.0
        return math.exp(-(new_cost - current_cost) / self.params.temperature)
    
    def optimize(self, initial_solution: Solution) -> Tuple[Solution, Dict]:
        """
        Виконує оптимізацію розв'язку адаптивним методом імітації відпалу.
        
        Returns:
            Tuple[Solution, Dict]: оптимальний розв'язок та статистика
        """
        self._initialize_stats()
        
        current_solution = deepcopy(initial_solution)
        best_solution = deepcopy(current_solution)
        
        current_cost = current_solution.compute_total_cost()
        best_cost = current_cost
        
        iteration = 0
        consecutive_stagnation = 0
        
        while iteration < self.max_iterations:
            # Генеруємо сусідній розв'язок
            new_solution, operation = generate_neighbor(current_solution)
            new_cost = new_solution.compute_total_cost()
            
            # Обчислюємо ймовірність прийняття
            acceptance_prob = self._acceptance_probability(current_cost, new_cost)
            
            # Вирішуємо, чи приймати новий розв'язок
            accepted = False
            improved = False
            
            if random.random() < acceptance_prob:
                accepted = True
                if new_cost < current_cost:
                    improved = True
                
                current_solution = new_solution
                current_cost = new_cost
                
                # Оновлюємо найкращий розв'язок
                if new_cost < best_cost:
                    best_solution = deepcopy(new_solution)
                    best_cost = new_cost
                    print(f"Iteration {iteration}: Found better solution with {operation} "
                          f"operation. New cost: {best_cost:.2f}")
            
            # Оновлюємо статистику
            self.params.update_rates(accepted, improved)
            
            # Перевіряємо стагнацію
            stagnation = self.params.check_stagnation(current_cost)
            if stagnation:
                consecutive_stagnation += 1
                if consecutive_stagnation >= self.max_stagnation_iterations:
                    print(f"Stopping due to prolonged stagnation after {iteration} iterations")
                    break
                self.params.handle_stagnation()
            else:
                consecutive_stagnation = 0
            
            # Адаптуємо параметри
            self.params.adapt_temperature()
            
            # Зберігаємо статистику
            self._update_stats(iteration, current_cost, best_cost, stagnation)
            
            iteration += 1
        
        print(f"\nAdaptive Simulated Annealing finished after {iteration} iterations")
        print(f"Initial cost: {initial_solution.compute_total_cost():.2f}")
        print(f"Final cost: {best_cost:.2f}")
        print(f"Improvement: {(initial_solution.compute_total_cost() - best_cost):.2f} "
              f"({(initial_solution.compute_total_cost() - best_cost) / initial_solution.compute_total_cost() * 100:.1f}%)")
        
        return best_solution, self.stats

def visualize_adaptive_sa_stats(stats: Dict):
    """Візуалізує статистику роботи адаптивного алгоритму."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Графік температури
    ax1.plot(stats['iterations'], stats['temperature'], label='Temperature')
    ax1.set_yscale('log')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Temperature')
    ax1.set_title('Adaptive Temperature')
    ax1.grid(True)
    
    # 2. Графік вартості розв'язку
    ax2.plot(stats['iterations'], stats['current_cost'], 
             label='Current Cost', alpha=0.5)
    ax2.plot(stats['iterations'], stats['best_cost'], 
             label='Best Cost', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Solution Cost')
    ax2.set_title('Solution Cost over Iterations')
    ax2.grid(True)
    ax2.legend()
    
    # 3. Графік рівнів прийняття
    ax3.plot(stats['iterations'], stats['acceptance_rate'], 
             label='Acceptance Rate')
    ax3.plot(stats['iterations'], stats['improvement_rate'], 
             label='Improvement Rate')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Rate')
    ax3.set_title('Acceptance and Improvement Rates')
    ax3.grid(True)
    ax3.legend()
    
    # 4. Графік подій стагнації
    stagnation_events = np.array(stats['stagnation_events']).astype(int)
    ax4.plot(stats['iterations'], stagnation_events, 
             label='Stagnation Events', drawstyle='steps-post')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Stagnation')
    ax4.set_title('Stagnation Events')
    ax4.grid(True)
    
    plt.tight_layout()
    return fig

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
    
    # Налаштовуємо та запускаємо адаптивну імітацію відпалу
    adaptive_sa = AdaptiveSimulatedAnnealing(
        initial_temp=100.0,
        max_iterations=5000,
        max_stagnation_iterations=500
    )
    
    improved_solution, stats = adaptive_sa.optimize(initial_solution)
    
    # Візуалізуємо результати
    fig1 = visualize_solution(initial_solution, "Початковий розв'язок (жадібний)")
    fig2 = visualize_solution(improved_solution, "Покращений розв'язок (Adaptive SA)")
    fig3 = visualize_adaptive_sa_stats(stats)
    plt.show()
