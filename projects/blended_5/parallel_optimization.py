from multiprocessing import Pool, cpu_count
from typing import List, Dict, Tuple, NamedTuple
import itertools
import time
from dataclasses import asdict
import pandas as pd
import seaborn as sns
from logistics_base import LogisticsProblem
from greedy_solution import Solution, greedy_solution
from simulated_annealing import SimulatedAnnealing, SimulatedAnnealingStats

class ExperimentResult(NamedTuple):
    """Результати одного експерименту."""
    params: Dict
    initial_cost: float
    final_cost: float
    improvement: float
    improvement_percent: float
    runtime: float
    iterations: int
    acceptance_rate: float
    improvement_rate: float

def run_single_experiment(args: Tuple[Solution, Dict]) -> ExperimentResult:
    """
    Запускає один експеримент з заданими параметрами.
    
    Args:
        args: кортеж (початковий_розв'язок, параметри)
    """
    initial_solution, params = args
    start_time = time.time()
    
    # Створюємо і запускаємо SA з заданими параметрами
    sa = SimulatedAnnealing(**params)
    final_solution, stats = sa.optimize(initial_solution)
    
    runtime = time.time() - start_time
    initial_cost = initial_solution.compute_total_cost()
    final_cost = final_solution.compute_total_cost()
    improvement = initial_cost - final_cost
    
    return ExperimentResult(
        params=params,
        initial_cost=initial_cost,
        final_cost=final_cost,
        improvement=improvement,
        improvement_percent=(improvement / initial_cost * 100),
        runtime=runtime,
        iterations=stats.iterations,
        acceptance_rate=stats.acceptance_rate(),
        improvement_rate=stats.improvement_rate()
    )

class ParallelOptimizer:
    """Клас для паралельного запуску оптимізації з різними параметрами."""
    
    def __init__(self, problem: LogisticsProblem):
        self.problem = problem
        self.initial_solution = None
        self.results: List[ExperimentResult] = []
        
    def generate_parameter_grid(self) -> List[Dict]:
        """Генерує сітку параметрів для експериментів."""
        param_grid = {
            'initial_temp': [50.0, 100.0, 200.0],
            'final_temp': [0.1, 1.0, 5.0],
            'alpha': [0.95, 0.98, 0.99],
            'iterations_per_temp': [50, 100],
            'max_iterations': [5000]
        }
        
        # Створюємо всі можливі комбінації параметрів
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))
        
        return [dict(zip(keys, combo)) for combo in combinations]
    
    def run_experiments(self, n_processes: int = None) -> pd.DataFrame:
        """
        Запускає експерименти паралельно.
        
        Args:
            n_processes: кількість процесів (за замовчуванням - кількість ядер)
        """
        if n_processes is None:
            n_processes = cpu_count()
            
        if self.initial_solution is None:
            self.initial_solution = greedy_solution(self.problem)
            
        # Генеруємо параметри
        param_grid = self.generate_parameter_grid()
        print(f"Running {len(param_grid)} experiments using {n_processes} processes...")
        
        # Готуємо аргументи для кожного експерименту
        args = [(deepcopy(self.initial_solution), params) 
                for params in param_grid]
        
        # Запускаємо експерименти паралельно
        with Pool(n_processes) as pool:
            self.results = pool.map(run_single_experiment, args)
            
        # Перетворюємо результати в DataFrame
        df_results = self._results_to_dataframe()
        
        return df_results
    
    def _results_to_dataframe(self) -> pd.DataFrame:
        """Перетворює результати в pandas DataFrame."""
        records = []
        for result in self.results:
            record = {
                'initial_cost': result.initial_cost,
                'final_cost': result.final_cost,
                'improvement': result.improvement,
                'improvement_percent': result.improvement_percent,
                'runtime': result.runtime,
                'iterations': result.iterations,
                'acceptance_rate': result.acceptance_rate,
                'improvement_rate': result.improvement_rate,
                **result.params  # розпаковуємо параметри
            }
            records.append(record)
            
        return pd.DataFrame(records)

def visualize_experiment_results(df: pd.DataFrame):
    """Візуалізує результати експериментів."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Покращення в залежності від початкової температури
    sns.boxplot(x='initial_temp', y='improvement_percent', data=df, ax=ax1)
    ax1.set_title('Improvement vs Initial Temperature')
    ax1.set_ylabel('Improvement (%)')
    
    # 2. Час виконання в залежності від кількості ітерацій
    sns.scatterplot(x='iterations', y='runtime', 
                   hue='alpha', size='initial_temp',
                   data=df, ax=ax2)
    ax2.set_title('Runtime vs Iterations')
    
    # 3. Теплова карта середнього покращення
    pivot_temp = df.pivot_table(
        values='improvement_percent',
        index='alpha',
        columns='initial_temp',
        aggfunc='mean'
    )
    sns.heatmap(pivot_temp, annot=True, fmt='.1f', ax=ax3)
    ax3.set_title('Average Improvement (%) by Parameters')
    
    # 4. Гістограма розподілу покращення
    sns.histplot(data=df, x='improvement_percent', bins=20, ax=ax4)
    ax4.set_title('Distribution of Improvements')
    ax4.set_xlabel('Improvement (%)')
    
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
    
    # Створюємо оптимізатор
    optimizer = ParallelOptimizer(problem)
    
    # Запускаємо експерименти
    results_df = optimizer.run_experiments()
    
    # Виводимо найкращі результати
    best_result = results_df.loc[results_df['improvement_percent'].idxmax()]
    print("\nНайкращий результат:")
    print(f"Покращення: {best_result['improvement_percent']:.1f}%")
    print(f"Час виконання: {best_result['runtime']:.1f} сек")
    print("\nПараметри:")
    for param in ['initial_temp', 'final_temp', 'alpha', 'iterations_per_temp']:
        print(f"{param}: {best_result[param]}")
    
    # Візуалізуємо результати
    fig = visualize_experiment_results(results_df)
    plt.show()
