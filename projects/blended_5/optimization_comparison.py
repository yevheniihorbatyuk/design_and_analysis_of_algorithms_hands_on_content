from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from time import time
from copy import deepcopy

@dataclass
class OptimizationMetrics:
    """Метрики якості розв'язку."""
    total_distance: float
    max_route_length: float
    avg_route_length: float
    std_route_length: float
    total_load: float
    load_utilization: float
    num_routes: int
    unassigned_orders: int
    computation_time: float
    num_iterations: int
    
    @classmethod
    def calculate(cls, solution: Solution, computation_time: float,
                 num_iterations: int) -> 'OptimizationMetrics':
        """Обчислює всі метрики для розв'язку."""
        # Загальна відстань
        total_distance = solution.compute_total_cost()
        
        # Метрики маршрутів
        route_lengths = []
        total_load = 0
        total_capacity = 0
        
        for route in solution.routes:
            # Довжина маршруту
            length = 0
            if route.orders:
                # Від складу до першої точки
                length += solution.problem.distance(route.warehouse,
                                                 route.orders[0].delivery_point)
                
                # Між точками маршруту
                for i in range(len(route.orders) - 1):
                    length += solution.problem.distance(
                        route.orders[i].delivery_point,
                        route.orders[i + 1].delivery_point
                    )
                
                # Від останньої точки до складу
                length += solution.problem.distance(
                    route.orders[-1].delivery_point,
                    route.warehouse
                )
            
            route_lengths.append(length)
            total_load += route.truck.current_load
            total_capacity += route.truck.capacity
        
        return cls(
            total_distance=total_distance,
            max_route_length=max(route_lengths) if route_lengths else 0,
            avg_route_length=np.mean(route_lengths) if route_lengths else 0,
            std_route_length=np.std(route_lengths) if route_lengths else 0,
            total_load=total_load,
            load_utilization=total_load / total_capacity if total_capacity > 0 else 0,
            num_routes=len(solution.routes),
            unassigned_orders=len(solution.unassigned_orders),
            computation_time=computation_time,
            num_iterations=num_iterations
        )

class Optimizer(ABC):
    """Базовий клас для всіх оптимізаторів."""
    
    @abstractmethod
    def optimize(self, initial_solution: Solution) -> Tuple[Solution, Dict]:
        """Виконує оптимізацію."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Повертає назву оптимізатора."""
        pass

class HillClimbingOptimizer(Optimizer):
    """Hill Climbing оптимізатор."""
    
    def __init__(self, max_iterations: int = 1000, max_no_improve: int = 100):
        self.max_iterations = max_iterations
        self.max_no_improve = max_no_improve
    
    @property
    def name(self) -> str:
        return "Hill Climbing"
    
    def optimize(self, initial_solution: Solution) -> Tuple[Solution, Dict]:
        start_time = time()
        solution = hill_climbing(initial_solution, self.max_iterations,
                               self.max_no_improve)
        computation_time = time() - start_time
        
        metrics = OptimizationMetrics.calculate(solution, computation_time,
                                              self.max_iterations)
        return solution, asdict(metrics)

class SimulatedAnnealingOptimizer(Optimizer):
    """Звичайний SA оптимізатор."""
    
    def __init__(self, initial_temp: float = 100.0, final_temp: float = 1.0,
                 alpha: float = 0.98, max_iterations: int = 5000):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.alpha = alpha
        self.max_iterations = max_iterations
    
    @property
    def name(self) -> str:
        return "Simulated Annealing"
    
    def optimize(self, initial_solution: Solution) -> Tuple[Solution, Dict]:
        start_time = time()
        sa = SimulatedAnnealing(
            initial_temp=self.initial_temp,
            final_temp=self.final_temp,
            alpha=self.alpha,
            max_iterations=self.max_iterations
        )
        solution, stats = sa.optimize(initial_solution)
        computation_time = time() - start_time
        
        metrics = OptimizationMetrics.calculate(solution, computation_time,
                                              stats.iterations)
        return solution, asdict(metrics)

class AdaptiveSAOptimizer(Optimizer):
    """Адаптивний SA оптимізатор."""
    
    def __init__(self, initial_temp: float = 100.0,
                 max_iterations: int = 5000,
                 max_stagnation_iterations: int = 500):
        self.initial_temp = initial_temp
        self.max_iterations = max_iterations
        self.max_stagnation_iterations = max_stagnation_iterations
    
    @property
    def name(self) -> str:
        return "Adaptive SA"
    
    def optimize(self, initial_solution: Solution) -> Tuple[Solution, Dict]:
        start_time = time()
        adaptive_sa = AdaptiveSimulatedAnnealing(
            initial_temp=self.initial_temp,
            max_iterations=self.max_iterations,
            max_stagnation_iterations=self.max_stagnation_iterations
        )
        solution, stats = adaptive_sa.optimize(initial_solution)
        computation_time = time() - start_time
        
        metrics = OptimizationMetrics.calculate(solution, computation_time,
                                              len(stats['iterations']))
        return solution, asdict(metrics)

class OptimizationComparison:
    """Клас для порівняння різних методів оптимізації."""
    
    def __init__(self, problem: LogisticsProblem, optimizers: List[Optimizer]):
        self.problem = problem
        self.optimizers = optimizers
        self.initial_solution = None
        self.results = []
        
    def run_comparison(self, num_runs: int = 5) -> pd.DataFrame:
        """Запускає порівняння оптимізаторів."""
        if self.initial_solution is None:
            self.initial_solution = greedy_solution(self.problem)
        
        results = []
        for optimizer in self.optimizers:
            print(f"\nRunning {optimizer.name}...")
            for run in range(num_runs):
                print(f"Run {run + 1}/{num_runs}")
                solution, metrics = optimizer.optimize(deepcopy(self.initial_solution))
                metrics['optimizer'] = optimizer.name
                metrics['run'] = run + 1
                results.append(metrics)
        
        return pd.DataFrame(results)

def visualize_comparison(df: pd.DataFrame):
    """Візуалізує порівняння методів оптимізації."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Порівняння загальної відстані
    sns.boxplot(x='optimizer', y='total_distance', data=df, ax=ax1)
    ax1.set_title('Total Distance by Optimizer')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Порівняння часу обчислень
    sns.boxplot(x='optimizer', y='computation_time', data=df, ax=ax2)
    ax2.set_title('Computation Time by Optimizer')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Порівняння використання вантажопідйомності
    sns.boxplot(x='optimizer', y='load_utilization', data=df, ax=ax3)
    ax3.set_title('Load Utilization by Optimizer')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Порівняння кількості маршрутів
    sns.boxplot(x='optimizer', y='num_routes', data=df, ax=ax4)
    ax4.set_title('Number of Routes by Optimizer')
    ax4.tick_params(axis='x', rotation=45)
    
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
    
    # Створюємо оптимізатори
    optimizers = [
        HillClimbingOptimizer(max_iterations=1000),
        SimulatedAnnealingOptimizer(max_iterations=5000),
        AdaptiveSAOptimizer(max_iterations=5000)
    ]
    
    # Запускаємо порівняння
    comparison = OptimizationComparison(problem, optimizers)
    results_df = comparison.run_comparison(num_runs=5)
    
    # Виводимо агреговані результати
    print("\nАгреговані результати:")
    summary = results_df.groupby('optimizer').agg({
        'total_distance': ['mean', 'std'],
        'computation_time': ['mean', 'std'],
        'load_utilization': ['mean', 'std'],
        'num_routes': ['mean', 'std']
    }).round(2)
    print(summary)
    
    # Візуалізуємо результати
    fig = visualize_comparison(results_df)
    plt.show()
