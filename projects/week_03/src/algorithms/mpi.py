"""
Демонстрація MPI (Message Passing Interface)
Домен: Паралельна обробка метеорологічних даних з мережі станцій

Концепції MPI:
- Point-to-point communication (Send/Recv)
- Collective communication (Scatter, Gather, Reduce, Broadcast)
- Master-Worker pattern
- Паралельна обробка даних
"""

from mpi4py import MPI
import numpy as np
import time
from datetime import datetime, timedelta
import json


class WeatherStation:
    """Генератор даних з метеостанції"""
    
    def __init__(self, station_id: int, region: str):
        self.station_id = station_id
        self.region = region
        
    def generate_data(self, days: int = 30) -> dict:
        """Генерує дані за N днів"""
        np.random.seed(self.station_id)  # Для відтворюваності
        
        temperatures = np.random.normal(20, 5, days)  # °C
        humidity = np.random.uniform(30, 90, days)     # %
        precipitation = np.random.exponential(2, days) # мм
        
        return {
            'station_id': self.station_id,
            'region': self.region,
            'temperatures': temperatures.tolist(),
            'humidity': humidity.tolist(),
            'precipitation': precipitation.tolist(),
            'days': days
        }


def analyze_station_data(data: dict) -> dict:
    """Аналізує дані однієї станції (виконується паралельно)"""
    
    temps = np.array(data['temperatures'])
    humidity = np.array(data['humidity'])
    precip = np.array(data['precipitation'])
    
    # Симуляція складних обчислень
    time.sleep(0.5)
    
    analysis = {
        'station_id': data['station_id'],
        'region': data['region'],
        'avg_temp': float(np.mean(temps)),
        'max_temp': float(np.max(temps)),
        'min_temp': float(np.min(temps)),
        'temp_std': float(np.std(temps)),
        'avg_humidity': float(np.mean(humidity)),
        'total_precipitation': float(np.sum(precip)),
        'rainy_days': int(np.sum(precip > 1.0)),
        'heat_wave_days': int(np.sum(temps > 30))
    }
    
    return analysis


def master_process(comm, size):
    """
    Головний процес (rank 0)
    Координує розподіл роботи між worker-процесами
    """
    
    print("=" * 70)
    print("🌍 СИСТЕМА АНАЛІЗУ МЕТЕОРОЛОГІЧНИХ ДАНИХ (MPI)")
    print("=" * 70)
    print(f"\n📡 Запущено {size} процесів (1 master + {size-1} workers)\n")
    
    # Створюємо дані з метеостанцій
    stations = [
        WeatherStation(1, "Київ"),
        WeatherStation(2, "Львів"),
        WeatherStation(3, "Одеса"),
        WeatherStation(4, "Харків"),
        WeatherStation(5, "Дніпро"),
        WeatherStation(6, "Запоріжжя"),
        WeatherStation(7, "Івано-Франківськ"),
        WeatherStation(8, "Чернівці"),
        WeatherStation(9, "Полтава"),
        WeatherStation(10, "Суми"),
    ]
    
    print(f"📊 Генеруємо дані з {len(stations)} метеостанцій...")
    station_data = [station.generate_data(days=30) for station in stations]
    print(f"✅ Згенеровано {len(station_data)} наборів даних\n")
    
    # ==========================================
    # 1. BROADCAST - Відправка конфігурації всім процесам
    # ==========================================
    config = {
        'analysis_type': 'full',
        'days': 30,
        'threshold_temp': 30
    }
    print("📢 Broadcast: Відправка конфігурації всім worker-процесам...")
    comm.bcast(config, root=0)
    
    # ==========================================
    # 2. SCATTER - Розподіл даних між worker-процесами
    # ==========================================
    print(f"📦 Scatter: Розподіл даних між {size-1} worker-процесами...")
    
    # Підготовка чанків для розподілу
    # Процес 0 (master) не обробляє дані, тому None для нього
    chunks = [None]  # для master
    chunk_size = len(station_data) // (size - 1)
    
    for i in range(1, size):
        start_idx = (i - 1) * chunk_size
        if i == size - 1:  # Останній процес бере всі залишки
            end_idx = len(station_data)
        else:
            end_idx = start_idx + chunk_size
        chunks.append(station_data[start_idx:end_idx])
    
    # Розподіляємо дані
    my_chunk = comm.scatter(chunks, root=0)
    
    # ==========================================
    # 3. GATHER - Збір результатів від worker-процесів
    # ==========================================
    print("⏳ Чекаємо на результати обробки від worker-процесів...\n")
    start_time = time.time()
    
    all_results = comm.gather(None, root=0)  # Master отримує None від себе
    
    processing_time = time.time() - start_time
    
    # Фільтруємо None і об'єднуємо результати
    results = []
    for result in all_results:
        if result is not None:
            results.extend(result)
    
    print("=" * 70)
    print("📈 РЕЗУЛЬТАТИ АНАЛІЗУ")
    print("=" * 70)
    
    # ==========================================
    # 4. REDUCE - Агрегація даних (температури)
    # ==========================================
    
    # Збираємо всі температури для глобального аналізу
    all_temps = []
    for r in results:
        all_temps.append(r['avg_temp'])
    
    print(f"\n🌡️  Аналіз за регіонами:\n")
    for result in results:
        print(f"   📍 {result['region']} (станція #{result['station_id']})")
        print(f"      • Середня t°: {result['avg_temp']:.1f}°C "
              f"(мін: {result['min_temp']:.1f}°C, макс: {result['max_temp']:.1f}°C)")
        print(f"      • Вологість: {result['avg_humidity']:.1f}%")
        print(f"      • Опади: {result['total_precipitation']:.1f}мм "
              f"({result['rainy_days']} дощових днів)")
        print(f"      • Спекотних днів (>30°C): {result['heat_wave_days']}")
        print()
    
    # Глобальна статистика
    print("🌍 Загальна статистика по всій території:")
    print(f"   • Середня температура: {np.mean(all_temps):.1f}°C")
    print(f"   • Найтепліший регіон: {results[np.argmax(all_temps)]['region']}")
    print(f"   • Найхолодніший регіон: {results[np.argmin(all_temps)]['region']}")
    
    total_precip = sum(r['total_precipitation'] for r in results)
    total_rainy_days = sum(r['rainy_days'] for r in results)
    
    print(f"   • Загальна кількість опадів: {total_precip:.1f}мм")
    print(f"   • Загальна кількість дощових днів: {total_rainy_days}")
    
    print(f"\n⏱️  Час обробки: {processing_time:.2f} секунд")
    print(f"💡 Паралельна обробка на {size-1} процесах")
    
    print("\n" + "=" * 70)


def worker_process(comm, rank, size):
    """
    Worker процес (rank > 0)
    Отримує дані і обробляє їх паралельно
    """
    
    # ==========================================
    # 1. BROADCAST - Отримання конфігурації
    # ==========================================
    config = comm.bcast(None, root=0)
    
    # ==========================================
    # 2. SCATTER - Отримання чанку даних
    # ==========================================
    my_chunk = comm.scatter(None, root=0)
    
    if my_chunk is not None:
        print(f"⚙️  Worker {rank}: Отримав {len(my_chunk)} станцій для обробки")
        
        # ==========================================
        # 3. ОБРОБКА ДАНИХ (паралельно на кожному процесі)
        # ==========================================
        my_results = []
        for station_data in my_chunk:
            result = analyze_station_data(station_data)
            my_results.append(result)
            print(f"   ✓ Worker {rank}: Оброблено станцію #{station_data['station_id']} "
                  f"({station_data['region']})")
        
        # ==========================================
        # 4. GATHER - Відправка результатів назад master-процесу
        # ==========================================
        comm.gather(my_results, root=0)
        print(f"📤 Worker {rank}: Відправив результати master-процесу\n")
    else:
        comm.gather(None, root=0)


def demonstrate_point_to_point(comm, rank, size):
    """
    Додаткова демонстрація: Point-to-point communication
    (Send/Recv між конкретними процесами)
    """
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("📨 ДЕМОНСТРАЦІЯ: Point-to-Point Communication")
        print("=" * 70 + "\n")
        
        # Master відправляє персональні повідомлення кожному worker
        for worker_rank in range(1, size):
            message = {
                'from': 0,
                'to': worker_rank,
                'task': f'Спеціальне завдання для worker {worker_rank}',
                'timestamp': datetime.now().isoformat()
            }
            comm.send(message, dest=worker_rank, tag=11)
            print(f"📤 Master -> Worker {worker_rank}: Відправлено повідомлення")
        
        # Отримання відповідей
        for worker_rank in range(1, size):
            response = comm.recv(source=worker_rank, tag=22)
            print(f"📥 Master <- Worker {worker_rank}: {response['message']}")
            
    else:
        # Worker отримує повідомлення
        message = comm.recv(source=0, tag=11)
        print(f"   📩 Worker {rank} отримав: {message['task']}")
        
        # Worker відправляє відповідь
        response = {
            'from': rank,
            'message': f'Завдання виконано! (Worker {rank})',
            'status': 'completed'
        }
        comm.send(response, dest=0, tag=22)


def main():
    """Головна функція з MPI"""
    
    # Ініціалізація MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Номер поточного процесу
    size = comm.Get_size()  # Загальна кількість процесів
    
    if size < 2:
        if rank == 0:
            print("❌ Помилка: Потрібно мінімум 2 процеси!")
            print("   Запустіть: mpiexec -n 4 python mpi_weather.py")
        return
    
    # Розподіл ролей
    if rank == 0:
        # MASTER процес
        master_process(comm, size)
        
        # Додаткова демонстрація
        demonstrate_point_to_point(comm, rank, size)
        
        print("\n✅ АНАЛІЗ ЗАВЕРШЕНО\n")
    else:
        # WORKER процеси
        worker_process(comm, rank, size)
        
        # Додаткова демонстрація
        demonstrate_point_to_point(comm, rank, size)


if __name__ == "__main__":
    main()