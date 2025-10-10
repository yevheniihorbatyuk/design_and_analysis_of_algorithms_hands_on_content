#!/usr/bin/env python3
"""
Модуль 8: Комплексна аналітична платформа
Використання: повний аналіз потоку даних з усіма алгоритмами одночасно
"""
import json
import time
from datetime import datetime, timezone
from collections import defaultdict
from confluent_kafka import Consumer, Producer
from src.data_structures.probabilistic import (
    BloomFilter,
    HyperLogLog,
    CountMinSketch,
    ReservoirSampling,
    MisraGries,
    MinHashLSH,
    SlidingWindow,
)

class ComprehensiveAnalyzer:
    """Комплексний аналізатор з усіма алгоритмами"""
    
    def __init__(
        self,
        bloom_capacity: int = 1_000_000,
        hll_precision: int = 14,
        cms_width: int = 8192,
        cms_depth: int = 5,
        reservoir_size: int = 1000,
        mg_k: int = 50,
        lsh_hashes: int = 128,
        window_size: int = 1000,
    ):
        # 1. Bloom Filter - дедуплікація
        self.bloom = BloomFilter.from_capacity(bloom_capacity, error_rate=0.001)
        
        # 2. HyperLogLog - кардинальність
        self.hll_vendors = HyperLogLog(precision=hll_precision)
        self.hll_pickup = HyperLogLog(precision=hll_precision)
        self.hll_dropoff = HyperLogLog(precision=hll_precision)
        self.hll_od_pairs = HyperLogLog(precision=hll_precision)
        self.hll_passengers = HyperLogLog(precision=hll_precision)
        
        # 3. Count-Min Sketch - частоти
        self.cms_routes = CountMinSketch(width=cms_width, depth=cms_depth)
        self.cms_pickup = CountMinSketch(width=cms_width, depth=cms_depth)
        
        # 4. Reservoir Sampling - репрезентативна вибірка
        self.reservoir = ReservoirSampling(k=reservoir_size)
        
        # 5. Misra-Gries - топ елементи
        self.mg_pickup = MisraGries(k=mg_k)
        self.mg_dropoff = MisraGries(k=mg_k)
        self.mg_routes = MisraGries(k=mg_k)
        
        # 6. LSH - схожі маршрути
        self.lsh = MinHashLSH(num_hashes=lsh_hashes, threshold=0.5)
        
        # 7. Sliding Window - метрики у вікні
        self.window_fare = SlidingWindow(window_size)
        self.window_distance = SlidingWindow(window_size)
        self.window_total = SlidingWindow(window_size)
        
        # Лічильники
        self.stats = {
            "total_events": 0,
            "unique_events": 0,
            "duplicates": 0,
            "total_fare": 0.0,
            "total_distance": 0.0,
        }
        
        self.start_time = time.time()
    
    def process_event(self, event: dict) -> dict:
        """Обробка події через всі алгоритми"""
        self.stats["total_events"] += 1
        
        # Витягуємо дані
        puloc = str(event.get("pulocation_id", "NA"))
        doloc = str(event.get("dolocation_id", "NA"))
        vendor = str(event.get("vendor", "NA"))
        passengers = str(event.get("passenger_count", "0"))
        fare = float(event.get("fare_amount", 0.0))
        distance = float(event.get("trip_distance", 0.0))
        total = float(event.get("total_amount", 0.0))
        
        route = f"{puloc}->{doloc}"
        
        # Формуємо ключ для дедуплікації
        bloom_key = (
            f"{event.get('pickup_ts', 'NA')}:"
            f"{vendor}:{puloc}:{doloc}:{distance}"
        )
        
        # 1. BLOOM FILTER - перевірка дублікату
        is_duplicate = bloom_key in self.bloom
        self.bloom.add(bloom_key)
        
        if is_duplicate:
            self.stats["duplicates"] += 1
            return {"duplicate": True}
        
        self.stats["unique_events"] += 1
        
        # 2. HYPERLOGLOG - унікальні значення
        self.hll_vendors.add(vendor)
        self.hll_pickup.add(puloc)
        self.hll_dropoff.add(doloc)
        self.hll_od_pairs.add(route)
        self.hll_passengers.add(passengers)
        
        # 3. COUNT-MIN SKETCH - частоти
        self.cms_routes.add(route)
        self.cms_pickup.add(puloc)
        
        # 4. RESERVOIR SAMPLING - зразки
        trip_sample = {
            "vendor": vendor,
            "puloc": puloc,
            "doloc": doloc,
            "fare": fare,
            "distance": distance,
        }
        self.reservoir.process_stream(iter([trip_sample]))
        
        # 5. MISRA-GRIES - топ елементи
        self.mg_pickup.process_stream(iter([puloc]))
        self.mg_dropoff.process_stream(iter([doloc]))
        self.mg_routes.process_stream(iter([route]))
        
        # 6. LSH - схожі маршрути
        features = set([
            f"PU:{puloc}",
            f"DO:{doloc}",
            f"V:{vendor}",
        ])
        self.lsh.add(route, features)
        
        # 7. SLIDING WINDOW - метрики у вікні
        self.window_fare.add(fare)
        self.window_distance.add(distance)
        self.window_total.add(total)
        
        # Агрегати
        self.stats["total_fare"] += fare
        self.stats["total_distance"] += distance
        
        return {"duplicate": False}
    
    def get_comprehensive_report(self) -> dict:
        """Генеруємо повний звіт"""
        runtime = time.time() - self.start_time
        
        # Top елементи з Misra-Gries
        top_pickup = sorted(
            self.mg_pickup.get_frequent_items().items(),
            key=lambda x: -x[1]
        )[:10]
        
        top_routes = sorted(
            self.mg_routes.get_frequent_items().items(),
            key=lambda x: -x[1]
        )[:10]
        
        # Вибірка з резервуару
        sample = self.reservoir.get_sample()
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": round(runtime, 2),
            
            # Загальна статистика
            "overview": {
                "total_events": self.stats["total_events"],
                "unique_events": self.stats["unique_events"],
                "duplicates_detected": self.stats["duplicates"],
                "dedup_rate": round(
                    self.stats["duplicates"] / self.stats["total_events"] * 100, 2
                ) if self.stats["total_events"] > 0 else 0,
                "throughput_eps": round(
                    self.stats["total_events"] / runtime, 2
                ) if runtime > 0 else 0,
            },
            
            # HyperLogLog - кардинальність
            "cardinality": {
                "unique_vendors": int(self.hll_vendors.estimate()),
                "unique_pickup_locations": int(self.hll_pickup.estimate()),
                "unique_dropoff_locations": int(self.hll_dropoff.estimate()),
                "unique_od_pairs": int(self.hll_od_pairs.estimate()),
                "unique_passenger_counts": int(self.hll_passengers.estimate()),
            },
            
            # Misra-Gries - топ елементи
            "top_locations": {
                "pickup": top_pickup,
                "routes": top_routes,
            },
            
            # Sliding Window - метрики
            "current_window": {
                "avg_fare": round(self.window_fare.get_average(), 2),
                "avg_distance": round(self.window_distance.get_average(), 2),
                "avg_total": round(self.window_total.get_average(), 2),
            },
            
            # Агрегати за весь час
            "totals": {
                "total_fare": round(self.stats["total_fare"], 2),
                "total_distance": round(self.stats["total_distance"], 2),
                "avg_fare_overall": round(
                    self.stats["total_fare"] / self.stats["unique_events"], 2
                ) if self.stats["unique_events"] > 0 else 0,
            },
            
            # Reservoir - вибірка
            "sample": {
                "size": len(sample),
                "examples": sample[:5],
            },
        }


def run_comprehensive_analyzer(
    brokers: str,
    topic_in: str,
    topic_out: str,
    report_interval: int = 15,
    group: str = "comprehensive-analyzer"
):
    """Запуск комплексного аналізатора"""
    consumer = Consumer({
        "bootstrap.servers": brokers,
        "group.id": group,
        "auto.offset.reset": "earliest",
    })
    
    producer = Producer({"bootstrap.servers": brokers})
    consumer.subscribe([topic_in])
    
    analyzer = ComprehensiveAnalyzer()
    last_report = time.time()
    
    print("🚀 COMPREHENSIVE ANALYZER STARTED")
    print("=" * 60)
    print(f"📥 Input topic: {topic_in}")
    print(f"📤 Output topic: {topic_out}")
    print(f"⏱️  Report interval: {report_interval}s")
    print(f"\n🧮 Active algorithms:")
    print("  1️⃣  Bloom Filter (deduplication)")
    print("  2️⃣  HyperLogLog (cardinality)")
    print("  3️⃣  Count-Min Sketch (frequencies)")
    print("  4️⃣  Reservoir Sampling (sampling)")
    print("  5️⃣  Misra-Gries (top-k items)")
    print("  6️⃣  LSH (similarity)")
    print("  7️⃣  Sliding Window (metrics)")
    print("=" * 60 + "\n")
    
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
                result = analyzer.process_event(event)
                
                # Періодичний звіт
                now = time.time()
                if now - last_report >= report_interval:
                    report = analyzer.get_comprehensive_report()
                    
                    print(f"\n{'='*60}")
                    print(f"📊 COMPREHENSIVE REPORT")
                    print(f"{'='*60}")
                    
                    print(f"\n📈 OVERVIEW:")
                    for key, value in report["overview"].items():
                        print(f"  • {key}: {value}")
                    
                    print(f"\n🔢 CARDINALITY (HyperLogLog):")
                    for key, value in report["cardinality"].items():
                        print(f"  • {key}: ~{value:,}")
                    
                    print(f"\n🏆 TOP PICKUP LOCATIONS (Misra-Gries):")
                    for loc, count in report["top_locations"]["pickup"][:5]:
                        print(f"  • Location {loc}: {count:,} trips")
                    
                    print(f"\n📊 CURRENT WINDOW (Sliding Window):")
                    for key, value in report["current_window"].items():
                        print(f"  • {key}: {value}")
                    
                    print(f"\n💰 TOTALS:")
                    for key, value in report["totals"].items():
                        print(f"  • {key}: {value}")
                    
                    print(f"\n{'='*60}\n")
                    
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
                import traceback
                traceback.print_exc()
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    finally:
        final_report = analyzer.get_comprehensive_report()
        print(f"\n{'='*60}")
        print("📈 FINAL REPORT")
        print(f"{'='*60}")
        print(json.dumps(final_report, indent=2))
        
        consumer.close()
        producer.flush()


if __name__ == "__main__":
    # Приклад використання:
    # python comprehensive_analytics.py
    run_comprehensive_analyzer(
        brokers="localhost:19092",
        topic_in="nyc.tlc.trips",
        topic_out="tlc.analytics.comprehensive",
        report_interval=15,
    )
