#!/usr/bin/env python3
"""
Модуль 2: HyperLogLog для підрахунку унікальних елементів
Використання: оцінка кардинальності в режимі реального часу
"""
import json
import time
from datetime import datetime, timezone
from collections import defaultdict
from confluent_kafka import Consumer, Producer
from src.data_structures.probabilistic import HyperLogLog

class CardinalityTracker:
    """Трекінг унікальних значень за різними вимірами"""
    
    def __init__(self, precision: int = 14):
        self.hlls = {
            "vendors": HyperLogLog(precision),
            "pickup_locations": HyperLogLog(precision),
            "dropoff_locations": HyperLogLog(precision),
            "od_pairs": HyperLogLog(precision),
            "payment_types": HyperLogLog(precision),
            "passenger_counts": HyperLogLog(precision),
        }
        self.events_processed = 0
        self.start_time = time.time()
    
    def process_event(self, event: dict):
        """Додаємо подію до всіх HLL структур"""
        self.events_processed += 1
        
        # Додаємо до відповідних HLL
        if vendor := event.get("vendor"):
            self.hlls["vendors"].add(str(vendor))
        
        if puloc := event.get("pulocation_id"):
            self.hlls["pickup_locations"].add(str(puloc))
        
        if doloc := event.get("dolocation_id"):
            self.hlls["dropoff_locations"].add(str(doloc))
        
        if puloc and doloc:
            od_pair = f"{puloc}->{doloc}"
            self.hlls["od_pairs"].add(od_pair)
        
        if payment := event.get("payment_type"):
            self.hlls["payment_types"].add(str(payment))
        
        if passengers := event.get("passenger_count"):
            self.hlls["passenger_counts"].add(str(passengers))
    
    def get_estimates(self) -> dict:
        """Отримуємо поточні оцінки кардинальності"""
        runtime = time.time() - self.start_time
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "events_processed": self.events_processed,
            "runtime_seconds": round(runtime, 2),
            "throughput_eps": round(self.events_processed / runtime, 2) if runtime > 0 else 0,
            "cardinality_estimates": {
                name: int(hll.estimate())
                for name, hll in self.hlls.items()
            }
        }
    
    def get_summary(self) -> dict:
        """Отримуємо детальну статистику"""
        estimates = self.get_estimates()
        card = estimates["cardinality_estimates"]
        
        return {
            **estimates,
            "insights": {
                "avg_od_pairs_per_pickup": (
                    round(card["od_pairs"] / card["pickup_locations"], 2)
                    if card["pickup_locations"] > 0 else 0
                ),
                "pickup_location_diversity": (
                    round(card["pickup_locations"] / self.events_processed, 3)
                    if self.events_processed > 0 else 0
                ),
                "od_pair_diversity": (
                    round(card["od_pairs"] / self.events_processed, 3)
                    if self.events_processed > 0 else 0
                ),
            }
        }


def run_cardinality_tracker(
    brokers: str,
    topic_in: str,
    topic_out: str,
    report_interval: int = 10,
    group: str = "hll-cardinality"
):
    """Запуск трекера кардинальності"""
    consumer = Consumer({
        "bootstrap.servers": brokers,
        "group.id": group,
        "auto.offset.reset": "earliest",
    })
    
    producer = Producer({"bootstrap.servers": brokers})
    consumer.subscribe([topic_in])
    
    tracker = CardinalityTracker()
    last_report = time.time()
    
    print(f"🚀 HyperLogLog Cardinality Tracker started")
    print(f"📥 Input: {topic_in}")
    print(f"📤 Output: {topic_out}")
    print(f"⏱️  Report interval: {report_interval}s\n")
    
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
                    summary = tracker.get_summary()
                    
                    print(f"\n📊 Cardinality Report")
                    print(f"  Events: {summary['events_processed']:,}")
                    print(f"  Throughput: {summary['throughput_eps']:.0f} events/sec")
                    print(f"\n  Unique Elements:")
                    for name, count in summary['cardinality_estimates'].items():
                        print(f"    • {name}: ~{count:,}")
                    
                    print(f"\n  Insights:")
                    for name, value in summary['insights'].items():
                        print(f"    • {name}: {value}")
                    
                    # Відправляємо в топік
                    producer.produce(
                        topic_out,
                        value=json.dumps(summary).encode(),
                    )
                    producer.poll(0)
                    
                    last_report = now
                    
            except json.JSONDecodeError:
                print(f"⚠️ Invalid JSON in message")
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    finally:
        summary = tracker.get_summary()
        print(f"\n📈 Final Summary:")
        print(json.dumps(summary, indent=2))
        consumer.close()
        producer.flush()


if __name__ == "__main__":
    # Приклад використання:
    # python hll_cardinality.py
    run_cardinality_tracker(
        brokers="localhost:19092",
        topic_in="nyc.tlc.trips",
        topic_out="tlc.cardinality.reports",
        report_interval=10,
    )
