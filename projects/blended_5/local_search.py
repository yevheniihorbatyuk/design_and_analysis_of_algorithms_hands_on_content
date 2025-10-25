from copy import deepcopy
from typing import List, Tuple, Optional, Order
import random
from logistics_base import LogisticsProblem, Order, Truck, Warehouse
from greedy_solution import Solution, Route, greedy_solution
from logistics_base import generate_test_problem


class LocalSearchOperator:
    """Базовий клас для операцій локального пошуку."""
    
    @staticmethod
    def swap_orders(solution: Solution, route1_idx: int, order1_idx: int,
                   route2_idx: int, order2_idx: int) -> bool:
        """
        Обмінює два замовлення між маршрутами.
        Повертає True, якщо обмін можливий і був виконаний.
        """
        if route1_idx == route2_idx:
            return False
            
        route1 = solution.routes[route1_idx]
        route2 = solution.routes[route2_idx]
        
        if order1_idx >= len(route1.orders) or order2_idx >= len(route2.orders):
            return False
            
        order1 = route1.orders[order1_idx]
        order2 = route2.orders[order2_idx]
        
        # Перевіряємо можливість обміну (вантажопідйомність)
        new_load1 = route1.truck.current_load - order1.volume + order2.volume
        new_load2 = route2.truck.current_load - order2.volume + order1.volume
        
        if (new_load1 <= route1.truck.capacity and 
            new_load2 <= route2.truck.capacity):
            # Виконуємо обмін
            route1.orders[order1_idx] = order2
            route2.orders[order2_idx] = order1
            
            # Оновлюємо завантаження
            route1.truck.current_load = new_load1
            route2.truck.current_load = new_load2
            return True
            
        return False
    
    @staticmethod
    def relocate_order(solution: Solution, from_route_idx: int, 
                      order_idx: int, to_route_idx: int,
                      new_position: int) -> bool:
        """
        Переміщує замовлення з одного маршруту в інший.
        """
        if from_route_idx == to_route_idx:
            return False
            
        from_route = solution.routes[from_route_idx]
        to_route = solution.routes[to_route_idx]
        
        if order_idx >= len(from_route.orders):
            return False
            
        order = from_route.orders[order_idx]
        
        # Перевіряємо можливість переміщення
        new_load = to_route.truck.current_load + order.volume
        if new_load <= to_route.truck.capacity:
            # Видаляємо з початкового маршруту
            from_route.orders.pop(order_idx)
            from_route.truck.current_load -= order.volume
            
            # Додаємо до нового маршруту
            to_route.orders.insert(min(new_position, len(to_route.orders)), order)
            to_route.truck.current_load += order.volume
            return True
            
        return False
    
    @staticmethod
    def two_opt_swap(solution: Solution, route_idx: int, i: int, j: int) -> bool:
        """
        Виконує 2-opt swap в межах одного маршруту (перевертає частину маршруту).
        """
        route = solution.routes[route_idx]
        if i >= j or i < 0 or j >= len(route.orders):
            return False
            
        # Перевертаємо частину маршруту між i та j
        route.orders[i:j+1] = reversed(route.orders[i:j+1])
        return True

def generate_neighbor(solution: Solution) -> Tuple[Solution, str]:
    """
    Генерує випадкового сусіда поточного розв'язку.
    Повертає (новий_розв'язок, тип_операції).
    """
    new_solution = deepcopy(solution)
    
    # Вибираємо випадкову операцію
    operation = random.choice(['swap', 'relocate', 'two_opt'])
    
    if operation == 'swap' and len(new_solution.routes) >= 2:
        # Вибираємо два різних маршрути
        route1_idx = random.randint(0, len(new_solution.routes) - 1)
        route2_idx = random.randint(0, len(new_solution.routes) - 2)
        if route2_idx >= route1_idx:
            route2_idx += 1
            
        if (len(new_solution.routes[route1_idx].orders) > 0 and
            len(new_solution.routes[route2_idx].orders) > 0):
            order1_idx = random.randint(0, len(new_solution.routes[route1_idx].orders) - 1)
            order2_idx = random.randint(0, len(new_solution.routes[route2_idx].orders) - 1)
            
            if LocalSearchOperator.swap_orders(new_solution, route1_idx, order1_idx,
                                            route2_idx, order2_idx):
                return new_solution, 'swap'
    
    elif operation == 'relocate' and len(new_solution.routes) >= 2:
        from_route_idx = random.randint(0, len(new_solution.routes) - 1)
        to_route_idx = random.randint(0, len(new_solution.routes) - 2)
        if to_route_idx >= from_route_idx:
            to_route_idx += 1
            
        if len(new_solution.routes[from_route_idx].orders) > 0:
            order_idx = random.randint(0, len(new_solution.routes[from_route_idx].orders) - 1)
            new_position = random.randint(0, len(new_solution.routes[to_route_idx].orders))
            
            if LocalSearchOperator.relocate_order(new_solution, from_route_idx,
                                                order_idx, to_route_idx, new_position):
                return new_solution, 'relocate'
    
    elif operation == 'two_opt':
        route_idx = random.randint(0, len(new_solution.routes) - 1)
        if len(new_solution.routes[route_idx].orders) >= 4:
            i = random.randint(0, len(new_solution.routes[route_idx].orders) - 2)
            j = random.randint(i + 1, len(new_solution.routes[route_idx].orders) - 1)
            
            if LocalSearchOperator.two_opt_swap(new_solution, route_idx, i, j):
                return new_solution, 'two_opt'
    
    # Якщо жодна операція не вдалася, повертаємо копію початкового розв'язку
    return new_solution, 'none'

def hill_climbing(initial_solution: Solution, 
                 max_iterations: int = 1000,
                 max_no_improve: int = 100) -> Solution:
    """
    Реалізує алгоритм Hill Climbing для покращення розв'язку.
    
    Args:
        initial_solution: початковий розв'язок
        max_iterations: максимальна кількість ітерацій
        max_no_improve: максимальна кількість ітерацій без покращення
    """
    current_solution = deepcopy(initial_solution)
    best_solution = deepcopy(current_solution)
    best_cost = current_solution.compute_total_cost()
    
    iterations = 0
    no_improve = 0
    
    while iterations < max_iterations and no_improve < max_no_improve:
        # Генеруємо сусіда
        neighbor, operation = generate_neighbor(current_solution)
        neighbor_cost = neighbor.compute_total_cost()
        
        # Якщо знайшли краще рішення
        if neighbor_cost < best_cost:
            best_solution = deepcopy(neighbor)
            best_cost = neighbor_cost
            current_solution = neighbor
            no_improve = 0
            print(f"Iteration {iterations}: Found better solution with {operation} "
                  f"operation. New cost: {best_cost:.2f}")
        else:
            no_improve += 1
            
        iterations += 1
    
    print(f"\nHill Climbing finished after {iterations} iterations")
    print(f"Initial cost: {initial_solution.compute_total_cost():.2f}")
    print(f"Final cost: {best_cost:.2f}")
    print(f"Improvement: {(initial_solution.compute_total_cost() - best_cost):.2f} "
          f"({(initial_solution.compute_total_cost() - best_cost) / initial_solution.compute_total_cost() * 100:.1f}%)")
    
    return best_solution

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
    
    # Покращуємо розв'язок за допомогою Hill Climbing
    improved_solution = hill_climbing(initial_solution)
    
    # Візуалізуємо початковий та покращений розв'язки
    fig1 = visualize_solution(initial_solution, "Початковий розв'язок (жадібний)")
    fig2 = visualize_solution(improved_solution, "Покращений розв'язок (Hill Climbing)")
    plt.show()
