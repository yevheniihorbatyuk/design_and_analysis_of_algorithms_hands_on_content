from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

@dataclass
class ProblemInstance:
    """Клас для опису конкретного екземпляру задачі."""
    name: str
    num_warehouses: int
    num_orders: int
    num_trucks: int
    area_size: float
    seed: int
    description: str

class ProblemGenerator:
    """Генератор різних типів тестових задач."""
    
    @staticmethod
    def generate_small_instance() -> ProblemInstance:
        """Генерує малу задачу."""
        return ProblemInstance(
            name="small",
            num_warehouses=2,
            num_orders=15,
            num_trucks=3,
            area_size=100,
            seed=42,
            description="Мала задача для швидкого тестування"
        )
    
    @staticmethod
    def generate_medium_instance() -> ProblemInstance:
        """Генерує середню задачу."""
        return ProblemInstance(
            name="medium",
            num_warehouses=3,
            num_orders=50,
            num_trucks=8,
            area_size=200,
            seed=42,
            description="Середня задача для основного тестування"
        )
    
    @staticmethod
    def generate_large_instance() -> ProblemInstance:
        """Генерує велику задачу."""
        return ProblemInstance(
            name="large",
            num_warehouses=5,
            num_orders=100,
            num_trucks=15,
            area_size=300,
            seed=42,
            description="Велика задача для стрес-тестування"
        )
    
    @staticmethod
    def generate_clustered_instance() -> ProblemInstance:
        """Генерує задачу з кластеризованими замовленнями."""
        return ProblemInstance(
            name="clustered",
            num_warehouses=3,
            num_orders=50,
            num_trucks=8,
            area_size=200,
            seed=43,
            description="Задача з кластеризованими замовленнями"
        )

class StatisticalAnalyzer:
    """Клас для статистичного аналізу результатів."""
    
    def __init__(self, results_df: pd.DataFrame):
        self.results_df = results_df
        self.significance_level = 0.05
    
    def perform_anova(self) -> Dict:
        """Виконує однофакторний дисперсійний аналіз."""
        groups = [group for _, group in self.results_df.groupby('optimizer')['total_distance']]
        f_statistic, p_value = stats.f_oneway(*groups)
        
        return {
            'test_name': 'ANOVA',
            'f_statistic': f_statistic,
            'p_value': p_value,
            'significant': p_value < self.significance_level
        }
    
    def perform_tukey_hsd(self) -> pd.DataFrame:
        """Виконує попарне порівняння методів (Tukey HSD)."""
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        
        tukey = pairwise_tukeyhsd(
            self.results_df['total_distance'],
            self.results_df['optimizer']
        )
        
        return pd.DataFrame(
            data=tukey._results_table.data[1:],
            columns=['group1', 'group2', 'meandiff', 'p_value', 'lower', 'upper', 'reject']
        )
    
    def compute_effect_sizes(self) -> pd.DataFrame:
        """Обчислює розміри ефекту (Cohen's d) для пар методів."""
        optimizers = self.results_df['optimizer'].unique()
        effect_sizes = []
        
        for i in range(len(optimizers)):
            for j in range(i + 1, len(optimizers)):
                opt1, opt2 = optimizers[i], optimizers[j]
                group1 = self.results_df[self.results_df['optimizer'] == opt1]['total_distance']
                group2 = self.results_df[self.results_df['optimizer'] == opt2]['total_distance']
                
                # Обчислення Cohen's d
                n1, n2 = len(group1), len(group2)
                var1, var2 = group1.var(), group2.var()
                pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
                cohens_d = (group1.mean() - group2.mean()) / pooled_se
                
                effect_sizes.append({
                    'optimizer1': opt1,
                    'optimizer2': opt2,
                    'cohens_d': cohens_d,
                    'effect_size': 'large' if abs(cohens_d) > 0.8 else
                                 'medium' if abs(cohens_d) > 0.5 else
                                 'small'
                })
        
        return pd.DataFrame(effect_sizes)

class TestingFramework:
    """Фреймворк для тестування оптимізаторів на різних задачах."""
    
    def __init__(self, optimizers: List[Optimizer]):
        self.optimizers = optimizers
        self.results: Dict[str, pd.DataFrame] = {}
        self.statistics: Dict[str, Dict] = {}
    
    def run_tests(self, num_runs: int = 5):
        """Запускає тести на різних типах задач."""
        problem_instances = [
            ProblemGenerator.generate_small_instance(),
            ProblemGenerator.generate_medium_instance(),
            ProblemGenerator.generate_large_instance(),
            ProblemGenerator.generate_clustered_instance()
        ]
        
        for instance in problem_instances:
            print(f"\nTesting {instance.name} instance...")
            problem = generate_test_problem(
                num_warehouses=instance.num_warehouses,
                num_orders=instance.num_orders,
                num_trucks=instance.num_trucks,
                area_size=instance.area_size,
                seed=instance.seed
            )
            
            comparison = OptimizationComparison(problem, self.optimizers)
            results_df = comparison.run_comparison(num_runs)
            
            # Зберігаємо результати
            self.results[instance.name] = results_df
            
            # Проводимо статистичний аналіз
            analyzer = StatisticalAnalyzer(results_df)
            self.statistics[instance.name] = {
                'anova': analyzer.perform_anova(),
                'tukey': analyzer.perform_tukey_hsd(),
                'effect_sizes': analyzer.compute_effect_sizes()
            }
    
    def generate_report(self, output_dir: str = 'results'):
        """Генерує звіт з результатами."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Створюємо візуалізації
        for instance_name, results_df in self.results.items():
            fig = self._create_instance_visualizations(
                instance_name, results_df, self.statistics[instance_name]
            )
            fig.savefig(output_path / f'results_{instance_name}_{timestamp}.png')
            plt.close(fig)
        
        # Зберігаємо статистику
        stats_summary = {}
        for instance_name, stats in self.statistics.items():
            stats_summary[instance_name] = {
                'anova': stats['anova'],
                'tukey': stats['tukey'].to_dict('records'),
                'effect_sizes': stats['effect_sizes'].to_dict('records')
            }
        
        with open(output_path / f'statistics_{timestamp}.json', 'w') as f:
            json.dump(stats_summary, f, indent=2)
    
    def _create_instance_visualizations(self, instance_name: str,
                                     results_df: pd.DataFrame,
                                     statistics: Dict) -> plt.Figure:
        """Створює візуалізації для конкретної задачі."""
        fig = plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(3, 2)
        
        # 1. Boxplot загальної відстані
        ax1 = fig.add_subplot(gs[0, :])
        sns.boxplot(x='optimizer', y='total_distance', data=results_df, ax=ax1)
        ax1.set_title(f'Total Distance Distribution ({instance_name} instance)')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Час обчислень
        ax2 = fig.add_subplot(gs[1, 0])
        sns.boxplot(x='optimizer', y='computation_time', data=results_df, ax=ax2)
        ax2.set_title('Computation Time')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Розміри ефекту
        ax3 = fig.add_subplot(gs[1, 1])
        effect_sizes = statistics['effect_sizes']
        sns.heatmap(
            effect_sizes.pivot(
                index='optimizer1',
                columns='optimizer2',
                values='cohens_d'
            ),
            annot=True,
            cmap='RdYlBu',
            ax=ax3
        )
        ax3.set_title("Effect Sizes (Cohen's d)")
        
        # 4. Статистична значущість
        ax4 = fig.add_subplot(gs[2, :])
        tukey_results = statistics['tukey']
        significant_pairs = tukey_results[tukey_results['reject']].copy()
        if not significant_pairs.empty:
            significant_pairs['p_value'] = -np.log10(
                significant_pairs['p_value'].astype(float)
            )
            sns.barplot(
                x='group1',
                y='p_value',
                hue='group2',
                data=significant_pairs,
                ax=ax4
            )
            ax4.set_title('Significant Differences (-log10 p-value)')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig

# Приклад використання:
if __name__ == "__main__":
    # Створюємо оптимізатори
    optimizers = [
        HillClimbingOptimizer(max_iterations=1000),
        SimulatedAnnealingOptimizer(max_iterations=5000),
        AdaptiveSAOptimizer(max_iterations=5000),
        GeneticAlgorithm(num_generations=100)
    ]
    
    # Створюємо фреймворк тестування
    framework = TestingFramework(optimizers)
    
    # Запускаємо тести
    framework.run_tests(num_runs=3)
    
    # Генеруємо звіт
    framework.generate_report()
    
    print("\nТестування завершено. Результати збережено в директорії 'results'.")
