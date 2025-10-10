#!/usr/bin/env python3
"""
Модуль 5: Misra-Gries для знаходження найчастіших елементів
Використання: пошук топ локацій та маршрутів без зберігання всього датасету
"""
import json
import time
from datetime import datetime, timezone
from confluent_kafka import Consumer, Producer
from src.data_structures.probabilistic import MisraGries

class FrequentItemsFinder:
    """Пошук найчастіших локацій, маршрутів, вендорів"""
    
    def __init__(self, k: int = 50):
        self.mg_pickup = MisraGries(k=k)
        self.mg_dropoff = MisraGries(k=k)
        self.mg_routes = MisraGries(k=k)
        self.mg_vendors = MisraGries(k=k)
        self.mg_hour = MisraGries(k=24)  # Години доби
        
        self.events_processed = 0
        self.start_time = time.time()
    
    def process_event(self, event: dict):
        """Обробляємо подію через всі MG структури"""
        self.events_processed += 1
        
        puloc = str(event.get("pulocation_id", "NA"))
        doloc = str(event.get("dolocation_id", "NA"))
        route = f"{puloc}->{doloc}"
        vendor = str(event.get("vendor", "NA"))
        
        # Час доби
        try:
            ts = event.get("pickup_ts", "")
            hour = datetime.fromisoformat(ts.replace("Z", "+00:00")).hour
            hour_str = f"{hour:02d}:00"
        except:
            hour_str = "NA"
        
        # Додаємо до MG структур
        self.mg_pickup.process_stream(iter([puloc]))
        self.mg_dropoff.process_stream(iter([doloc]))
        self.mg_routes.process_stream(iter([route]))
        self.mg_vendors.process_stream(iter([vendor]))
        self.mg_hour.process_stream(iter([hour_str]))
    
    def get_top_items(self, limit: int = 20) -> dict:
        """Отримуємо топ елементів по всіх категоріях"""
        return {
            "top_pickup_locations": sorted(
                self.mg_pickup.get_frequent_items().items(),
                key=lambda x: -x[1]
            )[:limit],
            "top_dropoff_locations": sorted(
                self.mg_dropoff.get_frequent_items().items(),
                key=lambda x: -x[1]
            )[:limit],
            "top_routes": sorted(
                self.mg_routes.get_frequent_items().items(),
                key=lambda x: -x[1]
            )[:limit],
            "top_vendors": sorted(
                self.mg_vendors.get_frequent_items().items(),
                key=lambda x: -x[1]
            ),
            "top_hours": sorted(
                self.mg_hour.get_frequent_items().items(),
                key=lambda x: -x[1]
            ),
        }
    
    def get_insights(self) -> dict:
        """Аналітичні інсайти"""
        top_items = self.get_top_items(limit=10)
        
        # Найпопулярніші локації
        top_pickup = top_items["top_pickup_locations"]
        top_dropoff = top_items["top_dropoff_locations"]
        
        # Розподіл по годинах
        hourly = dict(top_items["top_hours"])
        
        # Знаходимо пікову годину
        if hourly:
            peak_hour = max(hourly.items(), key=lambda x: x[1])
        else:
            peak_hour = ("NA", 0)
        
        return {
            "most_popular_pickup": top_pickup[0] if top_pickup else ("NA", 0),
            "most_popular_dropoff": top_dropoff[0] if top_dropoff else ("NA", 0),
            "most_popular_route": top_items["top_routes"][0] if top_items["top_routes"] else ("NA", 0),
            "peak_hour": peak_hour,
            "vendor_distribution": dict(top_items["top_vendors"]),
        }


def run_frequent_items_finder(
    brokers: str,
    topic_in: str,
    topic_out: str,
    k: int = 50,
    report_interval: int = 15,
    group: str = "misra-gries-finder"
):
    """Запуск пошуку частих елементів"""
    consumer = Consumer({
        "bootstrap.servers": brokers,
        "group.id": group,
        "auto.offset.reset": "earliest",
    })
    
    producer = Producer({"bootstrap.servers": brokers})
    consumer.subscribe([topic_in])
    
    finder = FrequentItemsFinder(k=k)
    last_report = time.time()
    
    print(f"🚀 Misra-Gries Frequent Items Finder started")
    print(f"📥 Input: {topic_in}")
    print(f"📤 Output: {topic_out}")
    print(f"🔢 k parameter: {k}\n")
    
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
                finder.process_event(event)
                
                # Періодичний звіт
                now = time.time()
                if now - last_report >= report_interval:
                    top_items = finder.get_top_items(limit=15)
                    insights = finder.get_insights()
                    
                    report = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "events_processed": finder.events_processed,
                        "top_items": top_items,
                        "insights": insights,
                    }
                    
                    print(f"\n📊 Frequent Items Report")
                    print(f"  Events: {finder.events_processed:,}")
                    
                    print(f"\n  🏆 Top 10 Pickup Locations:")
                    for i, (loc, count) in enumerate(top_items["top_pickup_locations"][:10], 1):
                        print(f"    {i}. Location {loc}: {count:,} trips")
                    
                    print(f"\n  🏆 Top 10 Routes:")
                    for i, (route, count) in enumerate(top_items["top_routes"][:10], 1):
                        print(f"    {i}. {route}: {count:,} trips")
                    
                    print(f"\n  ⏰ Peak Hour: {insights['peak_hour'][0]} ({insights['peak_hour'][1]:,} trips)")
                    print(f"  📍 Hottest Pickup: Location {insights['most_popular_pickup'][0]} ({insights['most_popular_pickup'][1]:,} trips)")
                    
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
        final_report = finder.get_top_items(limit=30)
        print(f"\n📈 Final Top Items:")
        print(json.dumps(final_report, indent=2))
        
        consumer.close()
        producer.flush()


if __name__ == "__main__":
    # Приклад використання:
    # python misra_gries.py
    run_frequent_items_finder(
        brokers="localhost:19092",
        topic_in="nyc.tlc.trips",
        topic_out="tlc.frequent.reports",
        k=50,
        report_interval=15,
    )
