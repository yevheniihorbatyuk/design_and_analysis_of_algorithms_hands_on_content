#!/usr/bin/env python3
"""
Модуль 3: Count-Min Sketch для трекінгу частот
Використання: оцінка частоти маршрутів без зберігання всіх даних
"""
import json
import time
from datetime import datetime, timezone
from collections import Counter
from confluent_kafka import Consumer, Producer
from src.data_structures.probabilistic import CountMinSketch

class RouteFrequencyTracker:
    """Трекінг частот маршрутів та локацій"""
    
    def __init__(self, width: int = 8192, depth: int = 5):
        self.cms_routes = CountMinSketch(width, depth)
        self.cms_pickup = CountMinSketch(width, depth)
        self.cms_dropoff = CountMinSketch(width, depth)
        self.cms_vendors = CountMinSketch(width, depth)
        
        # Для валідації точності (тримаємо точні значення для порівняння)
        self.exact_routes = Counter()
        self.exact_pickup = Counter()
        
        self.events_processed = 0
        self.start_time = time.time()
    
    def process_event(self, event: dict):
        """Додаємо подію до CMS структур"""
        self.events_processed += 1
        
        puloc = str(event.get("pulocation_id", "NA"))
        doloc = str(event.get("dolocation_id", "NA"))
        route = f"{puloc}->{doloc}"
        vendor = str(event.get("vendor", "NA"))
        
        # Додаємо до CMS
        self.cms_routes.add(route)
        self.cms_pickup.add(puloc)
        self.cms_dropoff.add(doloc)
        self.cms_vendors.add(vendor)
        
        # Також тримаємо точні значення для деяких категорій (для перевірки)
        self.exact_routes[route] += 1
        self.exact_pickup[puloc] += 1
    
    def query_route(self, puloc: str, doloc: str) -> int:
        """Запит частоти конкретного маршруту"""
        route = f"{puloc}->{doloc}"
        return self.cms_routes.estimate(route)
    
    def get_top_routes(self, k: int = 10) -> list:
        """Отримуємо топ-k маршрутів (використовуючи exact для пошуку, CMS для оцінки)"""
        top_routes = self.exact_routes.most_common(k)
        return [
            {
                "route": route,
                "cms_estimate": self.cms_routes.estimate(route),
                "exact_count": count,
                "error": abs(self.cms_routes.estimate(route) - count),
                "error_pct": abs(self.cms_routes.estimate(route) - count) / count * 100 if count > 0 else 0
            }
            for route, count in top_routes
        ]
    
    def get_top_pickups(self, k: int = 10) -> list:
        """Топ-k локацій пікапу"""
        top = self.exact_pickup.most_common(k)
        return [
            {
                "location_id": loc,
                "cms_estimate": self.cms_pickup.estimate(loc),
                "exact_count": count,
            }
            for loc, count in top
        ]
    
    def get_accuracy_metrics(self) -> dict:
        """Метрики точності CMS"""
        errors = []
        for route, exact in self.exact_routes.items():
            estimate = self.cms_routes.estimate(route)
            rel_error = abs(estimate - exact) / exact if exact > 0 else 0
            errors.append(rel_error)
        
        return {
            "avg_relative_error": sum(errors) / len(errors) if errors else 0,
            "max_relative_error": max(errors) if errors else 0,
            "routes_tracked": len(self.exact_routes),
        }


def run_frequency_tracker(
    brokers: str,
    topic_in: str,
    topic_out: str,
    report_interval: int = 15,
    group: str = "cms-frequency"
):
    """Запуск трекера частот"""
    consumer = Consumer({
        "bootstrap.servers": brokers,
        "group.id": group,
        "auto.offset.reset": "earliest",
    })
    
    producer = Producer({"bootstrap.servers": brokers})
    consumer.subscribe([topic_in])
    
    tracker = RouteFrequencyTracker()
    last_report = time.time()
    
    print(f"🚀 Count-Min Sketch Frequency Tracker started")
    print(f"📥 Input: {topic_in}")
    print(f"📤 Output: {topic_out}\n")
    
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"❌ Error: {msg.error()}")
                continue
            
            try:
                event = json.loads(msg.value())
                tracker.process_event(event)
                
                # Періодичний звіт
                now = time.time()
                if now - last_report >= report_interval:
                    top_routes = tracker.get_top_routes(10)
                    top_pickups = tracker.get_top_pickups(10)
                    accuracy = tracker.get_accuracy_metrics()
                    
                    report = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "events_processed": tracker.events_processed,
                        "top_routes": top_routes,
                        "top_pickup_locations": top_pickups,
                        "cms_accuracy": accuracy,
                    }
                    
                    print(f"\n📊 Frequency Report")
                    print(f"  Events: {tracker.events_processed:,}")
                    print(f"\n  Top 5 Routes:")
                    for i, r in enumerate(top_routes[:5], 1):
                        print(f"    {i}. {r['route']}: ~{r['cms_estimate']:,} trips (error: {r['error_pct']:.1f}%)")
                    
                    print(f"\n  CMS Accuracy:")
                    print(f"    • Avg relative error: {accuracy['avg_relative_error']:.2%}")
                    print(f"    • Max relative error: {accuracy['max_relative_error']:.2%}")
                    
                    # Відправляємо в топік
                    producer.produce(
                        topic_out,
                        value=json.dumps(report).encode(),
                    )
                    producer.poll(0)
                    
                    last_report = now
                    
            except json.JSONDecodeError:
                print(f"⚠️ Invalid JSON in message")
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    finally:
        consumer.close()
        producer.flush()


if __name__ == "__main__":
    # Приклад використання:
    # python cms_frequency.py
    run_frequency_tracker(
        brokers="localhost:19092",
        topic_in="nyc.tlc.trips",
        topic_out="tlc.frequency.reports",
        report_interval=15,
    )
