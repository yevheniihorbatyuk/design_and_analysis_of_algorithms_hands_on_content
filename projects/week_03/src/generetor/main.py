# services/generator/main.py

import time
import json
import random
from pathlib import Path
from typing import List, Dict # Додано Dict для тайпінгу в _send
from confluent_kafka import Producer
from config import GeneratorConfig, CITY_CONFIGS
from address_provider import AddressProvider
from generators import OrderGenerator, VehicleSimulator

class DataGenerator:
    """Main data generator orchestrator"""
    
    def __init__(self, config: GeneratorConfig):
        self.config = config
        
        # Setup address providers
        self.address_providers = {}
        for city in config.cities:
            self.address_providers[city.name] = AddressProvider(city)
        
        # Setup generators
        self.order_generators = {
            city.name: OrderGenerator(self.address_providers[city.name], city.name)
            for city in config.cities
        }
        
        # Setup vehicle simulators
        self.vehicles = []
        for city in config.cities:
            for i in range(config.vehicles_per_city):
                vehicle_id = f"{city.name[:3].upper()}-{i+1:03d}"
                driver_id = f"DRV-{random.randint(1000, 9999)}"
                simulator = VehicleSimulator(
                    vehicle_id, driver_id, city.name,
                    self.address_providers[city.name]
                )
                self.vehicles.append(simulator)
        
        # Setup output
        if config.output_mode == 'kafka':
            # Серіалізація даних буде виконана вручну при відправці.
            self.producer = Producer({
                'bootstrap.servers': config.kafka_broker,
                # Додаткові параметри, як-от клієнтський ID, можуть бути додані тут
            })
        else:
            self.output_dir = Path(config.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.file_handles = {
                'orders': open(self.output_dir / 'orders.jsonl', 'a'),
                'gps': open(self.output_dir / 'gps.jsonl', 'a'),
                'status': open(self.output_dir / 'status.jsonl', 'a')
            }
    
    def run(self):
        """Main generation loop"""
        print(f"Starting data generator...")
        print(f"  Cities: {[c.name for c in self.config.cities]}")
        print(f"  Orders/min: {self.config.orders_per_minute}")
        print(f"  Vehicles: {len(self.vehicles)}")
        print(f"  Output: {self.config.output_mode}")
        
        last_order_time = time.time()
        last_gps_time = time.time()
        
        order_interval = 60.0 / self.config.orders_per_minute
        gps_interval = self.config.gps_interval_seconds
        
        try:
            while True:
                now = time.time()
                
                # Generate orders
                if now - last_order_time >= order_interval:
                    self._generate_orders()
                    last_order_time = now
                
                # Update vehicles (GPS)
                if now - last_gps_time >= gps_interval:
                    self._update_vehicles()
                    last_gps_time = now
                
                # ЗМІНА 3: Рекомендовано викликати poll() для обробки черги асинхронних подій (доставки повідомлень)
                if self.config.output_mode == 'kafka':
                    self.producer.poll(0)
                
                time.sleep(0.1)  # Small sleep to prevent busy loop
        
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.cleanup()
    
    def _generate_orders(self):
        """Generate batch of orders"""
        # Apply peak hour multiplier
        current_hour = time.localtime().tm_hour
        multiplier = 1.0
        if self.config.peak_hours and current_hour in self.config.peak_hours:
            multiplier = self.config.peak_multiplier
        
        num_orders = int(multiplier)
        
        for _ in range(num_orders):
            # Random city
            city = random.choice(self.config.cities)
            generator = self.order_generators[city.name]
            
            order = generator.generate()
            self._send('orders', order.to_dict())
            
            # Maybe assign to vehicle
            if random.random() < 0.7:  # 70% immediate assignment
                available = [v for v in self.vehicles 
                             if v.status == 'available' and v.city_name == city.name]
                if available:
                    vehicle = random.choice(available)
                    vehicle.assign_order(order)
    
    def _update_vehicles(self):
        """Update all vehicles"""
        for vehicle in self.vehicles:
            gps_event, status_event = vehicle.update()
            
            self._send('gps_tracking', gps_event.to_dict())
            
            if status_event:
                self._send('vehicle_status', status_event.to_dict())
    
    def _send(self, topic: str, data: Dict):
        """Send data to Kafka or file"""
        if self.config.output_mode == 'kafka':
            TOPIC_PREFIX = "logistics." 
            full_topic = f"{TOPIC_PREFIX}{topic}"
            # ЗМІНА 4: Використання producer.produce() і серіалізація даних вручну
            # Confluent Kafka вимагає байтів, тому кодуємо JSON
            self.producer.produce(
                full_topic, 
                value=json.dumps(data).encode('utf-8')
                # Можна також додати key=... якщо потрібно
            )
        else:
            # Map to file handle
            file_key = 'orders' if topic == 'orders' else \
                     'gps' if topic == 'gps_tracking' else 'status'
            self.file_handles[file_key].write(json.dumps(data) + '\n')
            self.file_handles[file_key].flush()
    
    def cleanup(self):
        """Cleanup resources"""
        if self.config.output_mode == 'kafka':
            # ЗМІНА 5: Використання producer.flush() для очікування доставки всіх повідомлень
            # Метод close() не потрібен, flush() - достатньо
            print("Waiting for all messages to be delivered...")
            self.producer.flush()
        else:
            for f in self.file_handles.values():
                f.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate logistics data')
    parser.add_argument('--config', default='config.yaml', 
                        help='Path to config file')
    parser.add_argument('--cities', nargs='+', 
                        choices=list(CITY_CONFIGS.keys()),
                        help='Cities to use (overrides config)')
    parser.add_argument('--orders-per-min', type=int,
                        help='Orders per minute (overrides config)')
    parser.add_argument('--output', choices=['kafka', 'files'],
                        help='Output mode (overrides config)')
    
    args = parser.parse_args()
    
    # Load config
    if Path(args.config).exists():
        config = GeneratorConfig.from_yaml(args.config)
    else:
        # Default config
        config = GeneratorConfig(
            cities=[CITY_CONFIGS['kyiv']],
            orders_per_minute=50,
            vehicles_per_city=50
        )
    
    # Apply overrides
    if args.cities:
        config.cities = [CITY_CONFIGS[city] for city in args.cities]
    if args.orders_per_min:
        config.orders_per_minute = args.orders_per_min
    if args.output:
        config.output_mode = args.output
    
    # Run
    generator = DataGenerator(config)
    generator.run()


if __name__ == '__main__':
    main()