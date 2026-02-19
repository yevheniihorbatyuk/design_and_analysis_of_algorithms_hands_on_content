from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
from enum import Enum

@dataclass
class Point:
    """Представляє точку на карті (склад або точку доставки)."""
    id: int
    x: float
    y: float
    
@dataclass
class Order:
    """Представляє замовлення для доставки."""
    id: int
    delivery_point: Point
    volume: float  # об'єм замовлення
    
@dataclass
class Warehouse(Point):
    """Представляє склад."""
    capacity: float  # максимальна місткість складу
    
@dataclass
class Truck:
    """Представляє вантажівку."""
    id: int
    capacity: float  # вантажопідйомність
    current_load: float = 0.0  # поточне завантаження
    
class LogisticsProblem:
    """Клас, що представляє всю логістичну задачу."""
    
    def __init__(self, 
                 warehouses: List[Warehouse],
                 orders: List[Order],
                 trucks: List[Truck]):
        self.warehouses = warehouses
        self.orders = orders
        self.trucks = trucks
        self._distance_matrix = None
        
    def distance(self, point1: Point, point2: Point) -> float:
        """Обчислює евклідову відстань між двома точками."""
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def compute_distance_matrix(self) -> np.ndarray:
        """Обчислює матрицю відстаней між всіма точками."""
        if self._distance_matrix is None:
            # Створюємо список всіх точок (склади + точки доставки)
            all_points = self.warehouses + [order.delivery_point for order in self.orders]
            n = len(all_points)
            
            # Ініціалізуємо матрицю відстаней
            self._distance_matrix = np.zeros((n, n))
            
            # Заповнюємо матрицю відстаней
            for i in range(n):
                for j in range(i+1, n):
                    dist = self.distance(all_points[i], all_points[j])
                    self._distance_matrix[i,j] = dist
                    self._distance_matrix[j,i] = dist
                    
        return self._distance_matrix

def generate_test_problem(
    num_warehouses: int = 3,
    num_orders: int = 20,
    num_trucks: int = 5,
    area_size: float = 100.0,
    seed: int = 42
) -> LogisticsProblem:
    """
    Генерує тестовий екземпляр логістичної задачі.
    
    Args:
        num_warehouses: кількість складів
        num_orders: кількість замовлень
        num_trucks: кількість вантажівок
        area_size: розмір області (квадрат area_size x area_size)
        seed: seed для генератора випадкових чисел
    
    Returns:
        LogisticsProblem: згенерований екземпляр задачі
    """
    np.random.seed(seed)
    
    # Генеруємо склади
    warehouses = [
        Warehouse(
            id=i,
            x=np.random.uniform(0, area_size),
            y=np.random.uniform(0, area_size),
            capacity=np.random.uniform(1000, 2000)  # випадкова місткість складу
        )
        for i in range(num_warehouses)
    ]
    
    # Генеруємо точки доставки та замовлення
    orders = [
        Order(
            id=i,
            delivery_point=Point(
                id=i+num_warehouses,
                x=np.random.uniform(0, area_size),
                y=np.random.uniform(0, area_size)
            ),
            volume=np.random.uniform(10, 100)  # випадковий об'єм замовлення
        )
        for i in range(num_orders)
    ]
    
    # Генеруємо вантажівки з різною вантажопідйомністю
    trucks = [
        Truck(
            id=i,
            capacity=np.random.uniform(300, 500)  # випадкова вантажопідйомність
        )
        for i in range(num_trucks)
    ]
    
    return LogisticsProblem(warehouses, orders, trucks)

# Приклад використання:
if __name__ == "__main__":
    # Створюємо тестову задачу
    problem = generate_test_problem()
    
    # Обчислюємо матрицю відстаней
    distances = problem.compute_distance_matrix()
    
    print(f"Створено задачу з {len(problem.warehouses)} складами,",
          f"{len(problem.orders)} замовленнями та {len(problem.trucks)} вантажівками")
    print(f"\nРозмір матриці відстаней: {distances.shape}")
