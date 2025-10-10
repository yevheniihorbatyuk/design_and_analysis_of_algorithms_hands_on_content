# demo_actor_model_ray.py

"""
Демонстрація Моделі Акторів за допомогою Ray.

Задача: Створити систему, де "Майстер" асинхронно роздає завдання
незалежним "Працівникам", які виконують їх паралельно.

Ray - це сучасний фреймворк для розподілених обчислень на Python.
Клас, позначений декоратором `@ray.remote`, стає "актором".

Вимоги:
- pip install ray

Як запустити:
- python demo_actor_model_ray.py
"""
import ray
import time
import random

@ray.remote
class WorkerActor:
    """Актор-працівник, що виконує завдання."""
    def __init__(self, worker_id: int):
        self.id = worker_id
        print(f"[Worker {self.id}] створено.")

    def perform_task(self, task_id: int) -> str:
        """Симулює виконання довготривалого завдання."""
        print(f"[Worker {self.id}] Почав виконувати завдання {task_id}.")
        time.sleep(random.uniform(0.5, 2.0))
        result = f"Результат для завдання {task_id} від Worker {self.id}"
        print(f"[Worker {self.id}] Завершив завдання {task_id}.")
        return result

@ray.remote
class MasterActor:
    """Актор-майстер, що керує працівниками та роздає завдання."""
    def __init__(self, num_workers: int):
        print("[Master] створено.")
        # Створюємо інстанси акторів-працівників.
        # Кожен з них може виконуватися в окремому процесі.
        self.workers = [WorkerActor.remote(i) for i in range(num_workers)]
        print(f"[Master] Створено {num_workers} воркерів.")

    def assign_tasks(self, num_tasks: int) -> list:
        """Асинхронно роздає завдання воркерам."""
        print(f"[Master] Роздаємо {num_tasks} завдань...")
        results_refs = []
        for i in range(num_tasks):
            # Вибираємо випадкового воркера і викликаємо його метод.
            # Виклик `.remote()` не блокує виконання! Він миттєво повертає
            # "Future" або "Object Reference" - посилання на майбутній результат.
            worker = random.choice(self.workers)
            ref = worker.perform_task.remote(i)
            results_refs.append(ref)
        return results_refs

def run_ray_simulation(num_workers: int, num_tasks: int):
    """Запускає повну симуляцію."""
    
    # 1. Ініціалізація Ray
    # Ray автоматично запускає кластер на локальній машині.
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    
    print(f"Ray Dashboard доступний за адресою: {ray.get_dashboard_url()}")

    # 2. Створення актора-майстра
    master = MasterActor.remote(num_workers)

    # 3. Запуск симуляції
    start_time = time.perf_counter()
    
    # Майстер асинхронно роздає завдання
    # assign_tasks_ref - це теж посилання на майбутній результат (список посилань)
    assign_tasks_ref = master.assign_tasks.remote(num_tasks)
    
    # Отримуємо список посилань на результати
    results_refs = ray.get(assign_tasks_ref)
    
    print("\n[Main] Всі завдання роздано. Чекаємо на результати...")
    
    # 4. Отримання результатів
    # ray.get() блокує виконання, доки всі результати,
    # на які ми чекаємо, не будуть готові.
    results = ray.get(results_refs)
    
    total_time = (time.perf_counter() - start_time) * 1000

    print(f"\n[Main] Всі результати отримано за {total_time:.2f} мс.")
    print("\n--- Результати ---")
    for res in results:
        print(f"  - {res}")
        
    # 5. Завершення роботи Ray
    ray.shutdown()

if __name__ == "__main__":
    print("\n--- Демонстрація Моделі Акторів на Ray ---")
    run_ray_simulation(num_workers=4, num_tasks=15)