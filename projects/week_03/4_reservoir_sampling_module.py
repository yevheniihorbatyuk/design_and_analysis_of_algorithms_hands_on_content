#!/usr/bin/env python3
"""
Модуль 4: Reservoir Sampling для отримання репрезентативної вибірки
Використання: збір статистично репрезентативного сету поїздок
"""
import json
import time
import statistics
from datetime import datetime, timezone
from confluent_kafka import Consumer, Producer
from src.data_structures.probabilistic import ReservoirSampling

class TripSampler:
    """Отримання репрезентативної вибірки поїздок"""
    
    def __init__(self, sample_size: int = 1000):
        self.reservoir = ReservoirSampling(k=sample_size)
        self.events_seen = 0
        self.start_time = time.time()
    
    def process_event(self, event: dict):
        """Додаємо подію до резервуару"""
        self.events_seen += 1
        
        # Створюємо компактне представлення поїздки
        trip = {
            "ts": event.get("pickup_ts"),
            "vendor": event.get("vendor"),
            "puloc": event.get("pulocation_id"),
            "doloc": event.get("dolocation_id"),
            "passengers": event.get("passenger_count"),
            "distance": event.get("trip_distance"),
            "fare": event.get("fare_amount"),
            "total": event.get("total_amount"),
            "payment": event.get("payment_type"),
        }
        
        self.reservoir.process_stream(iter([trip]))
    
    def get_sample(self) -> list:
        """Отримуємо поточну вибірку"""
        return self.reservoir.get_sample()
    
    def get_sample_statistics(self) -> dict:
        """Статистика по вибірці"""
        sample = self.get_sample()
        
        if not sample:
            return {}
        
        # Збираємо числові метрики
        fares = [t["fare"] for t in sample if t.get("fare") is not None]
        distances = [t["distance"] for t in sample if t.get("distance") is not None]
        passengers = [t["passengers"] for t in sample if t.get("passengers") is not None]
        
        return {
            "sample_size": len(sample),
            "total_events_seen": self.events_seen,
            "sampling_rate": len(sample) / self.events_seen if self.events_seen > 0 else 0,
            "fare_stats": {
                "mean": round(statistics.mean(fares), 2) if fares else 0,
                "median": round(statistics.median(fares), 2) if fares else 0,
                "stdev": round(statistics.stdev(fares), 2) if len(fares) > 1 else 0,
                "min": round(min(fares), 2) if fares else 0,
                "max": round(max(fares), 2) if fares else 0,
            },
            "distance_stats": {
                "mean": round(statistics.mean(distances), 2) if distances else 0,
                "median": round(statistics.median(distances), 2) if distances else 0,
                "stdev": round(statistics.stdev(distances), 2) if len(distances) > 1 else 0,
            },
            "passenger_stats": {
                "mean": round(statistics.mean(passengers), 2) if passengers else 0,
                "median": statistics.median(passengers) if passengers else 0,
                "mode": statistics.mode(passengers) if passengers else 0,
            }
        }
    
    def analyze_sample(self) -> dict:
        """Детальний аналіз вибірки"""
        sample = self.get_sample()
        
        # Розподіл по вендорам
        vendor_dist = {}
        for trip in sample:
            vendor = str(trip.get("vendor", "NA"))
            vendor_dist[vendor] = vendor_dist.get(vendor, 0) + 1
        
        # Розподіл по кількості пасажирів
        passenger_dist = {}
        for trip in sample:
            passengers = str(trip.get("passengers", "NA"))
            passenger_dist[passengers] = passenger_dist.get(passengers, 0) + 1
        
        # Розподіл по типу оплати
        payment_dist = {}
        for trip in sample:
            payment = str(trip.get("payment", "NA"))
            payment_dist[payment] = payment_dist.get(payment, 0) + 1
        
        return {
            "vendor_distribution": vendor_dist,
            "passenger_distribution": passenger_dist,
            "payment_distribution": payment_dist,
        }


def run_sampler(
    brokers: str,
    topic_in: str,
    topic_out: str,
    sample_size: int = 1000,
    report_interval: int = 20,
    group: str = "reservoir-sampler"
):
    """Запуск семплера"""
    consumer = Consumer({
        "bootstrap.servers": brokers,
        "group.id": group,
        "auto.offset.reset": "earliest",
    })
    
    producer = Producer({"bootstrap.servers": brokers})
    consumer.subscribe([topic_in])
    
    sampler = TripSampler(sample_size=sample_size)
    last_report = time.time()
    
    print(f"🚀 Reservoir Sampler started")
    print(f"📥 Input: {topic_in}")
    print(f"📤 Output: {topic_out}")
    print(f"📊 Sample size: {sample_size}\n")
    
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
                sampler.process_event(event)
                
                # Періодичний звіт
                now = time.time()
                if now - last_report >= report_interval:
                    stats = sampler.get_sample_statistics()
                    analysis = sampler.analyze_sample()
                    sample = sampler.get_sample()
                    
                    report = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "statistics": stats,
                        "distributions": analysis,
                        "sample_trips": sample[:10],  # Перші 10 для прикладу
                    }
                    
                    print(f"\n📊 Sample Report")
                    print(f"  Events seen: {stats['total_events_seen']:,}")
                    print(f"  Sample size: {stats['sample_size']}")
                    print(f"  Sampling rate: {stats['sampling_rate']:.4%}")
                    
                    print(f"\n  Fare Statistics:")
                    print(f"    • Mean: ${stats['fare_stats']['mean']}")
                    print(f"    • Median: ${stats['fare_stats']['median']}")
                    print(f"    • Std Dev: ${stats['fare_stats']['stdev']}")
                    
                    print(f"\n  Distance Statistics:")
                    print(f"    • Mean: {stats['distance_stats']['mean']} mi")
                    print(f"    • Median: {stats['distance_stats']['median']} mi")
                    
                    print(f"\n  Distributions:")
                    print(f"    • Vendors: {analysis['vendor_distribution']}")
                    print(f"    • Passengers: {analysis['passenger_distribution']}")
                    
                    # Відправляємо в топік
                    producer.produce(
                        topic_out,
                        value=json.dumps(report).encode(),
                    )
                    producer.poll(0)
                    
                    last_report = now
                    
            except json.JSONDecodeError:
                print(f"⚠️ Invalid JSON in message")
            except Exception as e:
                print(f"⚠️ Error processing event: {e}")
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    finally:
        # Фінальний звіт
        final_stats = sampler.get_sample_statistics()
        print(f"\n📈 Final Statistics:")
        print(json.dumps(final_stats, indent=2))
        
        consumer.close()
        producer.flush()


if __name__ == "__main__":
    # Приклад використання:
    # python reservoir_sampling.py
    run_sampler(
        brokers="localhost:19092",
        topic_in="nyc.tlc.trips",
        topic_out="tlc.sample.reports",
        sample_size=1000,
        report_interval=20,
    )
