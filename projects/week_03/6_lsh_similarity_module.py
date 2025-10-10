#!/usr/bin/env python3
"""
Модуль 6: LSH (Locality-Sensitive Hashing) для пошуку схожих маршрутів
Використання: групування схожих поїздок, виявлення популярних кластерів
"""
import json
import time
from datetime import datetime, timezone
from collections import defaultdict, Counter
from confluent_kafka import Consumer, Producer
from src.data_structures.probabilistic import MinHashLSH

class SimilarRouteFinder:
    """Пошук схожих маршрутів через LSH"""
    
    def __init__(self, num_hashes: int = 128, threshold: float = 0.5):
        self.lsh = MinHashLSH(num_hashes=num_hashes, threshold=threshold)
        self.routes = {}  # {route_id: route_info}
        self.route_counter = 0
        self.events_processed = 0
        self.start_time = time.time()
    
    def process_event(self, event: dict):
        """Додаємо поїздку до LSH"""
        self.events_processed += 1
        
        puloc = str(event.get("pulocation_id", "NA"))
        doloc = str(event.get("dolocation_id", "NA"))
        vendor = str(event.get("vendor", "NA"))
        passengers = str(event.get("passenger_count", "0"))
        
        route_id = f"R{self.route_counter}"
        self.route_counter += 1
        
        # Створюємо set ознак для маршруту
        features = set([
            f"PU:{puloc}",
            f"DO:{doloc}",
            f"V:{vendor}",
            f"P:{passengers}",
        ])
        
        # Зберігаємо інформацію про маршрут
        self.routes[route_id] = {
            "puloc": puloc,
            "doloc": doloc,
            "vendor": vendor,
            "passengers": int(passengers) if passengers.isdigit() else 0,
            "fare": event.get("fare_amount"),
            "distance": event.get("trip_distance"),
        }
        
        # Додаємо до LSH
        self.lsh.add(route_id, features)
    
    def find_similar(self, puloc: str, doloc: str, vendor: str = None, passengers: str = None) -> list:
        """Знаходимо схожі маршрути"""
        query_features = set([f"PU:{puloc}", f"DO:{doloc}"])
        if vendor:
            query_features.add(f"V:{vendor}")
        if passengers:
            query_features.add(f"P:{passengers}")
        
        candidates = self.lsh.query(query_features)
        
        similar_routes = []
        for route_id in candidates:
            if route_id in self.routes:
                similar_routes.append({
                    "route_id": route_id,
                    **self.routes[route_id]
                })
        
        return similar_routes
    
    def get_clusters(self, min_cluster_size: int = 5) -> list:
        """Знаходимо кластери схожих маршрутів"""
        clusters = []
        processed = set()
        
        for route_id in list(self.routes.keys())[:1000]:  # Обмежуємо для продуктивності
            if route_id in processed:
                continue
            
            route = self.routes[route_id]
            features = set([
                f"PU:{route['puloc']}",
                f"DO:{route['doloc']}",
                f"V:{route['vendor']}",
            ])
            
            similar = self.lsh.query(features)
            
            if len(similar) >= min_cluster_size:
                cluster_routes = [self.routes[rid] for rid in similar if rid in self.routes]
                
                # Аналіз кластера
                pulocs = Counter(r["puloc"] for r in cluster_routes)
                dolocs = Counter(r["doloc"] for r in cluster_routes)
                
                clusters.append({
                    "size": len(cluster_routes),
                    "common_pickup": pulocs.most_common(1)[0] if pulocs else ("NA", 0),
                    "common_dropoff": dolocs.most_common(1)[0] if dolocs else ("NA", 0),
                    "route_ids": list(similar)[:10],  # Перші 10
                })
                
                processed.update(similar)
        
        return sorted(clusters, key=lambda x: -x["size"])
    
    def get_statistics(self) -> dict:
        """Статистика по LSH"""
        return {
            "events_processed": self.events_processed,
            "unique_routes": len(self.routes),
            "runtime_seconds": round(time.time() - self.start_time, 2),
        }


def run_similarity_finder(
    brokers: str,
    topic_in: str,
    topic_out: str,
    num_hashes: int = 128,
    threshold: float = 0.5,
    report_interval: int = 30,
    group: str = "lsh-similarity"
):
    """Запуск пошуку схожих маршрутів"""
    consumer = Consumer({
        "bootstrap.servers": brokers,
        "group.id": group,
        "auto.offset.reset": "earliest",
    })
    
    producer = Producer({"bootstrap.servers": brokers})
    consumer.subscribe([topic_in])
    
    finder = SimilarRouteFinder(num_hashes=num_hashes, threshold=threshold)
    last_report = time.time()
    
    print(f"🚀 LSH Similar Route Finder started")
    print(f"📥 Input: {topic_in}")
    print(f"📤 Output: {topic_out}")
    print(f"🔧 Num hashes: {num_hashes}, Threshold: {threshold}\n")
    
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
                    stats = finder.get_statistics()
                    clusters = finder.get_clusters(min_cluster_size=3)
                    
                    report = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "statistics": stats,
                        "clusters": clusters[:20],  # Топ 20 кластерів
                    }
                    
                    print(f"\n📊 Similarity Report")
                    print(f"  Events: {stats['events_processed']:,}")
                    print(f"  Unique routes: {stats['unique_routes']:,}")
                    print(f"  Clusters found: {len(clusters)}")
                    
                    if clusters:
                        print(f"\n  🔍 Top 5 Clusters:")
                        for i, cluster in enumerate(clusters[:5], 1):
                            print(f"    {i}. Size: {cluster['size']} trips")
                            print(f"       Common: {cluster['common_pickup'][0]} → {cluster['common_dropoff'][0]}")
                    
                    # Приклад пошуку схожих маршрутів
                    if finder.routes:
                        sample_route = list(finder.routes.values())[0]
                        similar = finder.find_similar(
                            sample_route["puloc"],
                            sample_route["doloc"],
                            sample_route["vendor"]
                        )
                        print(f"\n  📍 Example: {len(similar)} similar routes to "
                              f"{sample_route['puloc']}→{sample_route['doloc']}")
                    
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
        final_stats = finder.get_statistics()
        final_clusters = finder.get_clusters()
        
        print(f"\n📈 Final Statistics:")
        print(json.dumps(final_stats, indent=2))
        print(f"\n Total clusters: {len(final_clusters)}")
        
        consumer.close()
        producer.flush()


if __name__ == "__main__":
    # Приклад використання:
    # python lsh_similarity.py
    run_similarity_finder(
        brokers="localhost:19092",
        topic_in="nyc.tlc.trips",
        topic_out="tlc.similarity.reports",
        num_hashes=128,
        threshold=0.5,
        report_interval=30,
    )
