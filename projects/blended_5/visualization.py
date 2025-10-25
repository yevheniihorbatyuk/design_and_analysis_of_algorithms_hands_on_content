import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
import random
from logistics_base import LogisticsProblem, Point, Order, Warehouse
from greedy_solution import Solution, Route

def generate_distinct_colors(n: int) -> List[Tuple[float, float, float]]:
    """Генерує n візуально різних кольорів."""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + random.uniform(-0.2, 0.2)
        value = 0.7 + random.uniform(-0.2, 0.2)
        
        # Конвертуємо HSV в RGB
        h = hue * 6
        c = value * saturation
        x = c * (1 - abs(h % 2 - 1))
        m = value - c
        
        if h < 1:
            rgb = (c, x, 0)
        elif h < 2:
            rgb = (x, c, 0)
        elif h < 3:
            rgb = (0, c, x)
        elif h < 4:
            rgb = (0, x, c)
        elif h < 5:
            rgb = (x, 0, c)
        else:
            rgb = (c, 0, x)
            
        colors.append(tuple(v + m for v in rgb))
    
    return colors

def visualize_solution(solution: Solution, title: str = "Логістичне рішення"):
    """Візуалізує розв'язок логістичної задачі."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Генеруємо кольори для маршрутів
    route_colors = generate_distinct_colors(len(solution.routes))
    
    # Перший графік: маршрути на карті
    # -------------------------------
    
    # Відображаємо склади
    warehouse_xs = [w.x for w in solution.problem.warehouses]
    warehouse_ys = [w.y for w in solution.problem.warehouses]
    ax1.scatter(warehouse_xs, warehouse_ys, c='red', s=100, marker='s', 
               label='Склади')
    
    # Відображаємо точки доставки
    delivery_xs = [o.delivery_point.x for o in solution.problem.orders]
    delivery_ys = [o.delivery_point.y for o in solution.problem.orders]
    ax1.scatter(delivery_xs, delivery_ys, c='blue', s=50, 
               label='Точки доставки')
    
    # Відображаємо маршрути
    for route, color in zip(solution.routes, route_colors):
        if not route.orders:
            continue
            
        # Малюємо лінію від складу до першої точки
        ax1.plot([route.warehouse.x, route.orders[0].delivery_point.x],
                [route.warehouse.y, route.orders[0].delivery_point.y],
                c=color, alpha=0.5)
        
        # Малюємо лінії між точками маршруту
        for i in range(len(route.orders) - 1):
            current = route.orders[i].delivery_point
            next_point = route.orders[i + 1].delivery_point
            ax1.plot([current.x, next_point.x],
                    [current.y, next_point.y],
                    c=color, alpha=0.5)
        
        # Малюємо лінію від останньої точки назад до складу
        ax1.plot([route.orders[-1].delivery_point.x, route.warehouse.x],
                [route.orders[-1].delivery_point.y, route.warehouse.y],
                c=color, alpha=0.5)
    
    ax1.set_title("Маршрути доставки")
    ax1.legend()
    ax1.grid(True)
    
    # Другий графік: завантаження вантажівок
    # -------------------------------
    
    truck_loads = [(route.truck.current_load / route.truck.capacity * 100)
                   for route in solution.routes]
    truck_ids = [f"Truck {route.truck.id}" for route in solution.routes]
    
    bars = ax2.bar(truck_ids, truck_loads, color=route_colors)
    ax2.set_title("Завантаження вантажівок (%)")
    ax2.set_ylim(0, 100)
    ax2.grid(True)
    
    # Додаємо підписи значень над стовпцями
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    # Додаємо загальну інформацію
    plt.suptitle(title)
    total_cost = solution.compute_total_cost()
    unassigned = len(solution.unassigned_orders)
    fig.text(0.02, 0.02, 
             f'Загальна відстань: {total_cost:.2f}\n'
             f'Невиконані замовлення: {unassigned}',
             fontsize=10)
    
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
    
    # Знаходимо жадібний розв'язок
    solution = greedy_solution(problem)
    
    # Візуалізуємо розв'язок
    fig = visualize_solution(solution, "Жадібний алгоритм")
    
    # Показуємо графік
    plt.show()
