from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
import numpy as np
from logistics_base import LogisticsProblem, Order, Warehouse, Truck, Point

@dataclass
class Route:
    """Представляє маршрут для однієї вантажівки."""
    truck: Truck
    warehouse: Warehouse
    orders: List[Order] = field(default_factory=list)
    total_distance: float = 0.0
    
    def add_order(self, order: Order) -> bool:
        """Додає замовлення до маршруту, якщо це можливо."""
        if self.truck.current_load + order.volume <= self.truck.capacity:
            self.orders.append(order)
            self.truck.current_load += order.volume
            return True
        return False
    
    def __str__(self):
        return f"Route(truck={self.truck.id}, orders={[o.id for o in self.orders]}, " \
               f"load={self.truck.current_load:.1f}/{self.truck.capacity:.1f})"

class Solution:
    """Представляє розв'язок логістичної задачі."""
    
    def __init__(self, problem: LogisticsProblem):
        self.problem = problem
        self.routes: List[Route] = []
        self.unassigned_orders: Set[Order] = set(problem.orders)
        
    def add_route(self, route: Route):
        """Додає новий маршрут до розв'язку."""
        self.routes.append(route)
        self.unassigned_orders -= set(route.orders)
    
    def compute_total_cost(self) -> float:
        """Обчислює загальну вартість (відстань) розв'язку."""
        total_cost = 0.0
        for route in self.routes:
            if not route.orders:
                continue
                
            # Відстань від складу до першої точки
            total_cost += self.problem.distance(route.warehouse, route.orders[0].delivery_point)
            
            # Відстані між послідовними точками маршруту
            for i in range(len(route.orders) - 1):
                total_cost += self.problem.distance(
                    route.orders[i].delivery_point,
                    route.orders[i + 1].delivery_point
                )
            
            # Відстань від останньої точки назад до складу
            total_cost += self.problem.distance(
                route.orders[-1].delivery_point,
                route.warehouse
            )
            
        return total_cost

def greedy_solution(problem: LogisticsProblem) -> Solution:
    """
    Створює початковий розв'язок жадібним алгоритмом.
    
    Стратегія:
    1. Сортуємо замовлення за об'ємом (спочатку найбільші)
    2. Для кожного замовлення:
       - Знаходимо найближчий склад
       - Знаходимо вантажівку з достатньою вільною місткістю
       - Додаємо замовлення до маршруту цієї вантажівки
    """
    solution = Solution(problem)
    
    # Створюємо копію списку замовлень і сортуємо за об'ємом (спадання)
    sorted_orders = sorted(problem.orders, key=lambda x: x.volume, reverse=True)
    
    # Створюємо копію списку вантажівок
    available_trucks = problem.trucks.copy()
    
    for order in sorted_orders:
        # Знаходимо найближчий склад до точки доставки
        nearest_warehouse = min(
            problem.warehouses,
            key=lambda w: problem.distance(w, order.delivery_point)
        )
        
        # Шукаємо існуючий маршрут, який може взяти це замовлення
        route_found = False
        for route in solution.routes:
            if route.warehouse == nearest_warehouse and route.add_order(order):
                route_found = True
                solution.unassigned_orders.remove(order)
                break
        
        if not route_found and available_trucks:
            # Створюємо новий маршрут з новою вантажівкою
            truck = available_trucks.pop(0)
            new_route = Route(truck=truck, warehouse=nearest_warehouse)
            if new_route.add_order(order):
                solution.add_route(new_route)
    
    # Оптимізуємо кожен маршрут методом найближчого сусіда
    for route in solution.routes:
        if len(route.orders) > 1:
            route.orders = optimize_route_nearest_neighbor(
                route.warehouse, route.orders, problem
            )
    
    return solution

def optimize_route_nearest_neighbor(
    warehouse: Warehouse,
    orders: List[Order],
    problem: LogisticsProblem
) -> List[Order]:
    """
    Оптимізує порядок відвідування точок в маршруті методом найближчого сусіда.
    """
    current_point = warehouse
    unvisited = orders.copy()
    optimized_route = []
    
    while unvisited:
        # Знаходимо найближчу невідвідану точку
        next_order = min(
            unvisited,
            key=lambda o: problem.distance(current_point, o.delivery_point)
        )
        optimized_route.append(next_order)
        unvisited.remove(next_order)
        current_point = next_order.delivery_point
    
    return optimized_route

# Приклад використання:
if __name__ == "__main__":
    # Створюємо тестову задачу
    problem = generate_test_problem(
        num_warehouses=2,
        num_orders=10,
        num_trucks=3,
        seed=42
    )
    
    # Знаходимо жадібний розв'язок
    solution = greedy_solution(problem)
    
    # Виводимо результати
    print(f"\nЗнайдено розв'язок:")
    print(f"Загальна вартість: {solution.compute_total_cost():.2f}")
    print(f"Кількість маршрутів: {len(solution.routes)}")
    print(f"Невиконані замовлення: {len(solution.unassigned_orders)}")
    
    print("\nДеталі маршрутів:")
    for i, route in enumerate(solution.routes, 1):
        print(f"{i}. {route}")
