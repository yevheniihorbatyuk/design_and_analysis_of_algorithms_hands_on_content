#!/usr/bin/env python3
"""
Модуль 1: Bloom Filter для дедуплікації поїздок
Використання: виявлення дублікатів в режимі реального часу
"""
import json
import sys
from datetime import datetime, timezone
from confluent_kafka import Consumer, Producer
from src.data_structures.probabilistic import BloomFilter

class TripDeduplicator:
    """Дедуплікація поїздок за допомогою Bloom Filter"""
    
    def __init__(self, capacity: int = 1_000_000, error_rate: float = 0.001):
        self.bloom = BloomFilter.from_capacity(capacity, error_rate=error_rate)
        self.stats = {
            "total_events": 0,
            "duplicates_detected": 0,
            "unique_events": 0,
        }
    
    def make_key(self, event: dict) -> str:
        """Створюємо унікальний ключ для поїздки"""
        return (
            f"{event.get('pickup_ts', 'NA')}:"
            f"{event.get('vendor', 'NA')}:"
            f"{event.get('pulocation_id', 'NA')}:"
            f"{event.get('dolocation_id', 'NA')}:"
            f"{event.get('trip_distance', 'NA')}"
        )
    
    def process_event(self, event: dict) -> tuple[bool, dict]:
        """
        Обробка події
        Повертає: (is_duplicate, event_or_none)
        """
        self.stats["total_events"] += 1
        key = self.make_key(event)
        
        is_duplicate = key in self.bloom
        self.bloom.add(key)
        
        if is_duplicate:
            self.stats["duplicates_detected"] += 1
            return True, None
        else:
            self.stats["unique_events"] += 1
            return False, event
    
    def get_stats(self) -> dict:
        return {
            **self.stats,
            "dedup_rate": (
                self.stats["duplicates_detected"] / self.stats["total_events"]
                if self.stats["total_events"] > 0 else 0
            )
        }


def run_deduplicator(
    brokers: str,
    topic_in: str,
    topic_out: str,
    group: str = "bloom-dedup"
):
    """Запуск дедуплікатора"""
    consumer = Consumer({
        "bootstrap.servers": brokers,
        "group.id": group,
        "auto.offset.reset": "earliest",
    })
    
    producer = Producer({"bootstrap.servers": brokers})
    consumer.subscribe([topic_in])
    
    dedup = TripDeduplicator()
    
    print(f"🚀 Bloom Filter Deduplicator started")
    print(f"📥 Input: {topic_in}")
    print(f"📤 Output: {topic_out}")
    
    try:
        msg_count = 0
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"❌ Error: {msg.error()}")
                continue
            
            try:
                event = json.loads(msg.value())
                is_dup, clean_event = dedup.process_event(event)
                
                if not is_dup:
                    # Відправляємо унікальну подію
                    producer.produce(
                        topic_out,
                        key=msg.key(),
                        value=json.dumps(clean_event).encode(),
                    )
                
                msg_count += 1
                if msg_count % 1000 == 0:
                    stats = dedup.get_stats()
                    print(f"\n📊 Stats after {msg_count} messages:")
                    print(f"  Total: {stats['total_events']}")
                    print(f"  Unique: {stats['unique_events']}")
                    print(f"  Duplicates: {stats['duplicates_detected']}")
                    print(f"  Dedup rate: {stats['dedup_rate']:.2%}")
                    producer.poll(0)
                    
            except json.JSONDecodeError:
                print(f"⚠️ Invalid JSON in message")
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    finally:
        stats = dedup.get_stats()
        print(f"\n📈 Final stats:")
        print(json.dumps(stats, indent=2))
        consumer.close()
        producer.flush()


if __name__ == "__main__":
    # Приклад використання:
    # python bloom_dedup.py
    run_deduplicator(
        brokers="localhost:19092",
        topic_in="nyc.tlc.trips",
        topic_out="nyc.tlc.trips.deduped",
    )
