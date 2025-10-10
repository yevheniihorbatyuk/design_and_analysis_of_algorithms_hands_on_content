"""
Демонстрація Actor Model використовуючи Ray
Домен: Система управління рестораном

Актори:
- Chef (Кухар) - готує страви
- Waiter (Офіціант) - приймає замовлення
- OrderManager (Менеджер замовлень) - координує роботу
"""

import ray
import time
import random
from typing import List, Dict
from datetime import datetime


# Ініціалізація Ray
ray.init(ignore_reinit_error=True)


@ray.remote
class Chef:
    """Актор Кухаря - готує страви незалежно"""
    
    def __init__(self, name: str, specialty: str):
        self.name = name
        self.specialty = specialty
        self.orders_completed = 0
        self.is_busy = False
        
    def cook(self, dish: str, order_id: int) -> Dict:
        """Готує страву (симулює час приготування)"""
        self.is_busy = True
        cooking_time = random.uniform(1, 3)  # 1-3 секунди
        
        print(f"👨‍🍳 {self.name} почав готувати {dish} (замовлення #{order_id})")
        time.sleep(cooking_time)
        
        self.orders_completed += 1
        self.is_busy = False
        
        return {
            "chef": self.name,
            "dish": dish,
            "order_id": order_id,
            "cooking_time": round(cooking_time, 2),
            "completed_at": datetime.now().strftime("%H:%M:%S")
        }
    
    def get_status(self) -> Dict:
        """Повертає статус кухаря"""
        return {
            "name": self.name,
            "specialty": self.specialty,
            "is_busy": self.is_busy,
            "orders_completed": self.orders_completed
        }


@ray.remote
class Waiter:
    """Актор Офіціанта - приймає та доставляє замовлення"""
    
    def __init__(self, name: str):
        self.name = name
        self.orders_taken = 0
        
    def take_order(self, table_number: int, dishes: List[str]) -> Dict:
        """Приймає замовлення від столика"""
        self.orders_taken += 1
        order_id = self.orders_taken
        
        print(f"🙋 {self.name} прийняв замовлення від столика #{table_number}")
        print(f"   Страви: {', '.join(dishes)}")
        
        return {
            "order_id": order_id,
            "waiter": self.name,
            "table_number": table_number,
            "dishes": dishes,
            "ordered_at": datetime.now().strftime("%H:%M:%S")
        }
    
    def deliver_order(self, order_id: int, table_number: int):
        """Доставляє готове замовлення"""
        time.sleep(0.5)  # Час доставки
        print(f"✅ {self.name} доставив замовлення #{order_id} до столика #{table_number}")
        
    def get_stats(self) -> Dict:
        return {
            "name": self.name,
            "orders_taken": self.orders_taken
        }


@ray.remote
class OrderManager:
    """Актор Менеджера - координує роботу між офіціантами та кухарями"""
    
    def __init__(self):
        self.pending_orders = []
        self.completed_orders = []
        self.total_orders = 0
        
    def process_order(self, order: Dict, chefs: List) -> List:
        """Обробляє замовлення та розподіляє страви між кухарями"""
        self.total_orders += 1
        order_id = order["order_id"]
        
        print(f"\n📋 Менеджер обробляє замовлення #{order_id}")
        
        # Розподіляємо страви між доступними кухарями
        cooking_tasks = []
        for i, dish in enumerate(order["dishes"]):
            # Вибираємо кухаря (Round-robin)
            chef = chefs[i % len(chefs)]
            # Запускаємо приготування асинхронно
            task = chef.cook.remote(dish, order_id)
            cooking_tasks.append(task)
        
        return cooking_tasks
    
    def register_completed_order(self, order_id: int):
        """Реєструє завершене замовлення"""
        self.completed_orders.append(order_id)
        print(f"✨ Замовлення #{order_id} повністю готове!")
        
    def get_statistics(self) -> Dict:
        """Повертає статистику ресторану"""
        return {
            "total_orders": self.total_orders,
            "completed_orders": len(self.completed_orders),
            "pending_orders": len(self.pending_orders)
        }


# ============= СИМУЛЯЦІЯ РОБОТИ РЕСТОРАНУ =============

def run_restaurant_simulation():
    """Головна функція симуляції"""
    
    print("🍽️  РЕСТОРАН 'Actor Model' ВІДКРИВАЄТЬСЯ\n")
    print("=" * 60)
    
    # Створюємо акторів
    print("\n👥 Створюємо персонал...\n")
    
    # Офіціанти
    waiter1 = Waiter.remote("Олег")
    waiter2 = Waiter.remote("Марія")
    waiters = [waiter1, waiter2]
    
    # Кухарі
    chef1 = Chef.remote("Іван", "Гарячі страви")
    chef2 = Chef.remote("Ольга", "Салати")
    chef3 = Chef.remote("Петро", "Десерти")
    chefs = [chef1, chef2, chef3]
    
    # Менеджер замовлень
    manager = OrderManager.remote()
    
    print("✅ Персонал готовий до роботи!\n")
    print("=" * 60)
    
    # Симуляція замовлень від різних столиків
    orders = [
        {"table": 1, "dishes": ["Борщ", "Вареники", "Компот"]},
        {"table": 2, "dishes": ["Салат Цезар", "Стейк"]},
        {"table": 3, "dishes": ["Піца Маргарита", "Тірамісу", "Капучино"]},
        {"table": 4, "dishes": ["Суші сет", "Мисо суп"]},
    ]
    
    all_cooking_tasks = []
    
    # Обробка замовлень (асинхронно)
    for order_data in orders:
        # Випадковий офіціант приймає замовлення
        waiter = random.choice(waiters)
        order = ray.get(waiter.take_order.remote(
            order_data["table"], 
            order_data["dishes"]
        ))
        
        # Менеджер розподіляє страви між кухарями
        cooking_tasks = ray.get(manager.process_order.remote(order, chefs))
        all_cooking_tasks.extend(cooking_tasks)
        
        # Невелика пауза між замовленнями
        time.sleep(0.3)
    
    print("\n" + "=" * 60)
    print("⏳ Кухарі готують страви паралельно...")
    print("=" * 60 + "\n")
    
    # Чекаємо на завершення всіх страв (паралельна обробка!)
    completed_dishes = ray.get(all_cooking_tasks)
    
    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТИ РОБОТИ")
    print("=" * 60 + "\n")
    
    # Виводимо інформацію про приготовані страви
    print("🍽️  Приготовані страви:")
    for dish in completed_dishes:
        print(f"   • {dish['dish']} - готував {dish['chef']} "
              f"за {dish['cooking_time']}с (замовлення #{dish['order_id']})")
    
    # Статистика кухарів
    print("\n👨‍🍳 Статистика кухарів:")
    for chef in chefs:
        status = ray.get(chef.get_status.remote())
        print(f"   • {status['name']} ({status['specialty']}): "
              f"{status['orders_completed']} страв")
    
    # Статистика офіціантів
    print("\n🙋 Статистика офіціантів:")
    for waiter in waiters:
        stats = ray.get(waiter.get_stats.remote())
        print(f"   • {stats['name']}: {stats['orders_taken']} замовлень")
    
    # Загальна статистика
    manager_stats = ray.get(manager.get_statistics.remote())
    print(f"\n📈 Загальна статистика:")
    print(f"   • Всього замовлень: {manager_stats['total_orders']}")
    print(f"   • Всього страв приготовлено: {len(completed_dishes)}")
    
    print("\n" + "=" * 60)
    print("🎉 РОБОЧИЙ ДЕНЬ ЗАВЕРШЕНО!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        run_restaurant_simulation()
    finally:
        # Очищення Ray
        ray.shutdown()
        print("\n👋 Ray зупинено")