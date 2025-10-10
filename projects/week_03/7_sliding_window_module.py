#!/usr/bin/env python3
"""
Модуль 7: Sliding Window для обчислення метрик у рухомому вікні
Використання: метрики за останні N подій (throughput, середні значення)
"""
import json
import time
from datetime import datetime, timezone
from collections import deque
from confluent_kafka import Consumer, Producer
from src.data_structures.probabilistic import SlidingWindow

class SlidingWindowAnalyzer:
    """Аналіз метрик у рухомому вікні"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        
        # Окремі вікна для різних метрик
        self.fare_window = SlidingWindow(window_size)
        self.distance_window = SlidingWindow(window_size)
        self.tip_window = SlidingWindow(window_size)
        
        # Вікна для підрахунку подій
        self.events = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        
        self.events_processed = 0
        self.start_time = time.time()
    
    def process_event(self, event: dict):
        """Додаємо подію до вікна"""
        self.events_processed += 1
        current_time = time.time()
        
        # Додаємо метрики
        fare = float(event.get("fare_amount", 0.0))
        distance = float(event.get("trip_distance", 0.0))
        
        # Обчислюємо чайові (якщо є total_amount)
        total = float(event.get("total_amount", 0.0))
        tip = total - fare if total > fare else 0.0
        
        self.fare_window.add(fare)
        self.distance_window.add(distance)
        self.tip_window.add(tip)
        
        # Зберігаємо подію та час
        self.events.append(event)
        self.timestamps.append(current_time)
    
    def get_metrics(self) -> dict:
        """Отримуємо поточні метрики"""
        # Пропускна здатність (events per second)
        if len(self.timestamps) >= 2:
            time_span = self.timestamps[-1] - self.timestamps[0]
            throughput = len(self.timestamps) / time_span if time_span > 0 else 0
        else:
            throughput = 0
        
        # Розподіл по вендорам у вікні
        vendor_counts = {}
        for event in self.events:
            vendor = str(event.get("vendor", "NA"))
            vendor_counts[vendor] = vendor_counts.get(vendor, 0) + 1
        
        # Розподіл по кількості пасажирів
        passenger_counts = {}
        for event in self.events:
            passengers = str(event.get("passenger_count", "NA"))
            passenger_counts[passengers] = passenger_counts.get(passengers, 0) + 1
        
        return {
            "window_size": len(self.events),
            "max_window_size": self.window_size,
            "throughput_eps": round(throughput, 2),
            "averages": {
                "fare": round(self.fare_window.get_average(), 2),
                "distance": round(self.distance_window.get_average(), 2),
                "tip": round(self.tip_window.get_average(), 2),
                "tip_percentage": round(
                    (self.tip_window.get_average() / self.fare_window.get_average() * 100)
                    if self.fare_window.get_average() > 0 else 0,
                    2
                ),
            },
            "distributions": {
                "vendors": vendor_counts,
                "passengers": passenger_counts,
            }
        }
    
    def get_trends(self) -> dict:
        """Аналіз трендів (порівняння першої та другої половини вікна)"""
        if len(self.events) < 10:
            return {}
        
        mid = len(self.events) // 2
        
        # Перша половина
        first_half_fares = [
            float(e.get("fare_amount", 0))
            for e in list(self.events)[:mid]
        ]
        
        # Друга половина
        second_half_fares = [
            float(e.get("fare_amount", 0))
            for e in list(self.events)[mid:]
        ]
        
        avg_first = sum(first_half_fares) / len(first_half_fares) if first_half_fares else 0
        avg_second = sum(second_half_fares) / len(second_half_fares) if second_half_fares else 0
        
        fare_trend = "up" if avg_second > avg_first else "down" if avg_second < avg_first else "stable"
        fare_change = abs(avg_second - avg_first)
        fare_change_pct = (fare_change / avg_first * 100) if avg_first > 0 else 0
        
        return {
            "fare_trend": fare_trend,
            "fare_change": round(fare_change, 2),
            "fare_change_pct": round(fare_change_pct, 2),
            "avg_first_half": round(avg_first, 2),
            "avg_second_half": round(avg_second, 2),
        }


def run_sliding_window_analyzer(
    brokers: str,
    topic_in: str,
    topic_out: str,
    window_size: int = 1000,
    report_interval: int = 10,
    group: str = "sliding-window"
):
    """Запуск аналізатора рухомого вікна"""
    consumer = Consumer({
        "bootstrap.servers": brokers,
        "group.id": group,
        "auto.offset.reset": "earliest",
    })
    
    producer = Producer({"bootstrap.servers": brokers})
    consumer.subscribe([topic_in])
    
    analyzer = SlidingWindowAnalyzer(window_size=window_size)
    last_report = time.time()
    
    print(f"🚀 Sliding Window Analyzer started")
    print(f"📥 Input: {topic_in}")
    print(f"📤 Output: {topic_out}")
    print(f"🪟 Window size: {window_size} events\n")
    
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
                analyzer.process_event(event)
                
                # Періодичний звіт
                now = time.time()
                if now - last_report >= report_interval:
                    metrics = analyzer.get_metrics()
                    trends = analyzer.get_trends()
                    
                    report = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "total_events_processed": analyzer.events_processed,
                        "metrics": metrics,
                        "trends": trends,
                    }
                    
                    print(f"\n📊 Sliding Window Report")
                    print(f"  Total events: {analyzer.events_processed:,}")
                    print(f"  Window fill: {metrics['window_size']}/{metrics['max_window_size']}")
                    print(f"  Throughput: {metrics['throughput_eps']:.1f} events/sec")
                    
                    print(f"\n  📈 Averages (last {metrics['window_size']} events):")
                    print(f"    • Fare: ${metrics['averages']['fare']}")
                    print(f"    • Distance: {metrics['averages']['distance']} mi")
                    print(f"    • Tip: ${metrics['averages']['tip']} ({metrics['averages']['tip_percentage']}%)")
                    
                    if trends:
                        print(f"\n  📉 Trends:")
                        print(f"    • Fare trend: {trends['fare_trend']} "
                              f"({trends['avg_first_half']} → {trends['avg_second_half']})")
                        print(f"    • Change: {trends['fare_change_pct']:.1f}%")
                    
                    print(f"\n  📊 Vendor distribution:")
                    for vendor, count in sorted(metrics['distributions']['vendors'].items()):
                        pct = count / metrics['window_size'] * 100
                        print(f"    • Vendor {vendor}: {count} ({pct:.1f}%)")
                    
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
                print(f"⚠️ Error: {e}")
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    finally:
        final_metrics = analyzer.get_metrics()
        print(f"\n📈 Final Metrics:")
        print(json.dumps(final_metrics, indent=2))
        
        consumer.close()
        producer.flush()


if __name__ == "__main__":
    # Приклад використання:
    # python sliding_window.py
    run_sliding_window_analyzer(
        brokers="localhost:19092",
        topic_in="nyc.tlc.trips",
        topic_out="tlc.window.reports",
        window_size=1000,
        report_interval=10,
    )
