import random
import time
import json
import argparse
from typing import Generator, Dict, Any, List, Optional
import uuid

class DataStreamGenerator:
    """Генератор потоку даних для тестування алгоритмів обробки великих даних."""
    
    def __init__(self, 
                 data_type: str = "web_logs", 
                 rate: float = 1.0,
                 error_rate: float = 0.05,
                 duplicate_rate: float = 0.1):
        """
        Ініціалізує генератор потоку даних.
        
        Args:
            data_type: Тип даних для генерації ('web_logs', 'transactions', 'social_media', 'iot_sensors')
            rate: Кількість елементів в секунду
            error_rate: Ймовірність генерації помилкових даних
            duplicate_rate: Ймовірність дублікатів
        """
        self.data_type = data_type
        self.rate = rate
        self.error_rate = error_rate
        self.duplicate_rate = duplicate_rate
        self.last_item = None
        
        # Словник з раніше згенерованими елементами для створення дублікатів
        self.previous_items = []
        self.max_previous_items = 1000
        
        # Ініціалізація генераторів даних відповідно до типу
        if data_type == "web_logs":
            self.generate_item = self._generate_web_log
        elif data_type == "transactions":
            self.generate_item = self._generate_transaction
        elif data_type == "social_media":
            self.generate_item = self._generate_social_media
        elif data_type == "iot_sensors":
            self.generate_item = self._generate_iot_sensor
        else:
            raise ValueError(f"Невідомий тип даних: {data_type}")
    
    def stream(self) -> Generator[Dict[str, Any], None, None]:
        """Генерує нескінченний потік даних."""
        while True:
            # Визначення, чи генерувати дублікат
            if self.previous_items and random.random() < self.duplicate_rate:
                item = random.choice(self.previous_items)
            else:
                item = self.generate_item()
                if len(self.previous_items) >= self.max_previous_items:
                    self.previous_items.pop(0)
                self.previous_items.append(item)
            
            yield item
            self.last_item = item
            
            # Затримка для контролю швидкості генерації
            time.sleep(1.0 / self.rate)
    
    def _generate_web_log(self) -> Dict[str, Any]:
        """Генерує запис веб-логу."""
        ip_addresses = [f"192.168.1.{random.randint(1, 255)}", 
                       f"10.0.0.{random.randint(1, 255)}", 
                       f"172.16.{random.randint(1, 255)}.{random.randint(1, 255)}"]
        
        paths = ["/home", "/products", "/about", "/contact", "/login", "/register", 
                "/profile", "/cart", "/checkout", "/search", "/api/v1/users", 
                "/api/v1/products", "/admin"]
        
        methods = ["GET", "POST", "PUT", "DELETE"]
        status_codes = [200, 201, 204, 400, 401, 403, 404, 500]
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15",
            "Mozilla/5.0 (iPad; CPU OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15"
        ]
        
        # Додаємо деякі помилки для реалістичності
        if random.random() < self.error_rate:
            return {
                "timestamp": time.time(),
                "ip": random.choice(ip_addresses),
                "method": random.choice(methods),
                "path": random.choice(paths) + "?" + "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(5, 20))),
                "status": random.choice([400, 401, 403, 404, 500]),
                "response_time": random.uniform(0.5, 10.0),
                "user_agent": random.choice(user_agents),
                "error": True
            }
        
        return {
            "timestamp": time.time(),
            "ip": random.choice(ip_addresses),
            "method": random.choice(methods),
            "path": random.choice(paths),
            "status": random.choice(status_codes),
            "response_time": random.uniform(0.01, 2.0),
            "user_agent": random.choice(user_agents),
            "error": False
        }
    
    def _generate_transaction(self) -> Dict[str, Any]:
        """Генерує запис транзакції."""
        user_ids = [f"user_{i}" for i in range(1, 101)]
        product_ids = [f"product_{i}" for i in range(1, 201)]
        payment_methods = ["credit_card", "debit_card", "paypal", "bank_transfer", "crypto"]
        
        if random.random() < self.error_rate:
            return {
                "transaction_id": str(uuid.uuid4()),
                "timestamp": time.time(),
                "user_id": random.choice(user_ids),
                "product_id": random.choice(product_ids),
                "amount": random.uniform(1.0, 1000.0),
                "currency": random.choice(["USD", "EUR", "GBP", "JPY"]),
                "payment_method": random.choice(payment_methods),
                "status": "failed"
            }
        
        return {
            "transaction_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "user_id": random.choice(user_ids),
            "product_id": random.choice(product_ids),
            "amount": random.uniform(1.0, 1000.0),
            "currency": random.choice(["USD", "EUR", "GBP", "JPY"]),
            "payment_method": random.choice(payment_methods),
            "status": "success"
        }
    
    def _generate_social_media(self) -> Dict[str, Any]:
        """Генерує запис соціальних медіа."""
        user_ids = [f"user_{i}" for i in range(1, 1001)]
        
        actions = ["post", "like", "comment", "share", "follow", "unfollow"]
        hashtags = ["#технології", "#програмування", "#python", "#bigdata", "#алгоритми", 
                    "#навчання", "#університет", "#студенти", "#дослідження", "#наука"]
        
        content_lengths = [0, 10, 50, 100, 500, 1000, 5000]
        
        return {
            "event_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "user_id": random.choice(user_ids),
            "action": random.choice(actions),
            "content_length": random.choice(content_lengths),
            "hashtags": random.sample(hashtags, random.randint(0, min(5, len(hashtags)))),
            "device": random.choice(["mobile", "desktop", "tablet"]),
            "location": random.choice(["Україна", "США", "Німеччина", "Індія", "Японія", "Австралія"])
        }
    
    def _generate_iot_sensor(self) -> Dict[str, Any]:
        """Генерує запис з IoT-датчика."""
        sensor_ids = [f"sensor_{i}" for i in range(1, 51)]
        sensor_types = ["temperature", "humidity", "pressure", "motion", "light", "gas", "water"]
        
        if random.random() < self.error_rate:
            return {
                "sensor_id": random.choice(sensor_ids),
                "timestamp": time.time(),
                "sensor_type": random.choice(sensor_types),
                "value": None,
                "battery": random.uniform(0.0, 100.0),
                "error_code": random.randint(1, 10)
            }
        
        value_ranges = {
            "temperature": (-20.0, 50.0),
            "humidity": (0.0, 100.0),
            "pressure": (900.0, 1100.0),
            "motion": (0, 1),
            "light": (0.0, 1000.0),
            "gas": (0.0, 500.0),
            "water": (0.0, 100.0)
        }
        
        sensor_type = random.choice(sensor_types)
        min_val, max_val = value_ranges[sensor_type]
        
        return {
            "sensor_id": random.choice(sensor_ids),
            "timestamp": time.time(),
            "sensor_type": sensor_type,
            "value": random.uniform(min_val, max_val),
            "battery": random.uniform(0.0, 100.0),
            "error_code": None
        }

def save_stream_to_file(generator: DataStreamGenerator, filename: str, count: int = 1000):
    """Зберігає згенеровані дані у файл."""
    with open(filename, 'w', encoding='utf-8') as f:
        stream = generator.stream()
        for _ in range(count):
            item = next(stream)
            f.write(json.dumps(item) + '\n')

def print_stream(generator: DataStreamGenerator, count: Optional[int] = None):
    """Виводить потік даних у консоль."""
    stream = generator.stream()
    i = 0
    try:
        while count is None or i < count:
            item = next(stream)
            print(json.dumps(item, ensure_ascii=False))
            i += 1
    except KeyboardInterrupt:
        print("\nГенерацію потоку даних зупинено.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Генератор потоку даних для тестування алгоритмів.')
    parser.add_argument('--type', default='web_logs', choices=['web_logs', 'transactions', 'social_media', 'iot_sensors'],
                       help='Тип даних для генерації')
    parser.add_argument('--rate', type=float, default=1.0,
                       help='Кількість елементів у секунду')
    parser.add_argument('--count', type=int, default=None,
                       help='Кількість елементів для генерації (None для нескінченного потоку)')
    parser.add_argument('--output', type=str, default=None,
                       help='Файл для збереження даних (якщо не вказано, виводить у консоль)')
    parser.add_argument('--error-rate', type=float, default=0.05,
                       help='Ймовірність помилкових даних')
    parser.add_argument('--duplicate-rate', type=float, default=0.1,
                       help='Ймовірність дублікатів')
    
    args = parser.parse_args()
    
    generator = DataStreamGenerator(
        data_type=args.type,
        rate=args.rate,
        error_rate=args.error_rate,
        duplicate_rate=args.duplicate_rate
    )
    
    if args.output:
        save_stream_to_file(generator, args.output, args.count or 1000)
        print(f"Дані збережено у файл {args.output}")
    else:
        print_stream(generator, args.count)