import random
import json
from typing import List, Dict, Optional
from pathlib import Path

# ЗМІНА: Використовуємо асинхронні бібліотеки
import asyncio
import aiohttp
from config import CityConfig


class AddressProvider:
    """Provides real addresses for cities using Nominatim API"""
    
    def __init__(self, city: CityConfig, cache_dir: str = "./data/cache"):
        self.city = city
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / f"{city.name.lower()}_addresses.json"
        
        # Викликаємо синхронний метод, який внутрішньо запускає асинхронний цикл
        self.addresses = self._load_or_fetch_addresses()
    
    def _load_or_fetch_addresses(self) -> List[Dict]:
        """Load from cache or fetch from Nominatim"""
        if self.cache_file.exists():
            print(f"Завантаження адрес із кешу: {self.cache_file}")
            try:
                with open(self.cache_file, encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Помилка завантаження кешу: {e}. Виконую нову вибірку.")
        
        print(f"Вибірка адрес для {self.city.name} (Використовуючи асинхронний клієнт з обмеженням 1 зап/сек)...")
        
        # ЗМІНА: Запуск асинхронної функції у синхронному середовищі
        addresses = asyncio.run(self._async_fetch_addresses(count=1500))
        
        # Save to cache
        if addresses:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(addresses, f, indent=2, ensure_ascii=False)
        else:
            print(f"Попередження: Не вдалося отримати адреси для {self.city.name}.")
        
        return addresses
    
    async def _async_fetch_addresses(self, count: int = 1500) -> List[Dict]:
        """Fetch real addresses from Nominatim (OpenStreetMap) asynchronously"""
        addresses = []
        
        queries = [
            'restaurant', 'cafe', 'shop', 'office', 'hotel',
            'school', 'hospital', 'bank', 'pharmacy', 'supermarket',
            'kindergarten', 'apartment', 'atm'
        ]
        
        limit_per_query = max(50, int(count / len(queries) * 1.5))
        
        print(f"  Ціль: ~{count} унікальних адрес ({limit_per_query} на тип запиту).")

        # Асинхронний клієнт сесії
        headers = {'User-Agent': 'LogisticsProject/1.0 (Contact: user@example.com)'}
        async with aiohttp.ClientSession(headers=headers) as session:
            for query in queries:
                # Ранній вихід, якщо мета досягнута
                if len(addresses) >= count:
                    print(f"  Загальна кількість адрес ({len(addresses)}) досягла цілі, зупиняю вибірку.")
                    break
                
                # ВАЖЛИВО: Асинхронна затримка для дотримання політики Nominatim (1 запит/сек)
                await asyncio.sleep(1) 
                
                try:
                    url = "https://nominatim.openstreetmap.org/search"
                    params = {
                        'q': f"{query} {self.city.name}",
                        'format': 'json',
                        'limit': limit_per_query,
                        'viewbox': f"{self.city.bounds['min_lon']},{self.city.bounds['min_lat']},"
                                   f"{self.city.bounds['max_lon']},{self.city.bounds['max_lat']}",
                        'bounded': 1
                    }
                    
                    # Виконання асинхронного GET-запиту
                    async with session.get(url, params=params) as response:
                        response.raise_for_status()
                        results = await response.json()
                        
                        for result in results:
                            address = {
                                'lat': float(result['lat']),
                                'lon': float(result['lon']),
                                'display_name': result.get('display_name', 'Невідома адреса'),
                                'type': result.get('type', 'unknown'),
                                'zone': self._get_zone(float(result['lat']), float(result['lon']))
                            }
                            addresses.append(address)
                        
                        print(f"  Отримано {len(results)} нових адрес для '{query}'. Поточний загал: {len(addresses)}")
                        
                except aiohttp.client_exceptions.ClientResponseError as he:
                    print(f"  HTTP Помилка вибірки {query}: {he.message}. Статус: {he.status}")
                    if he.status == 429:
                        print("  429 Забагато запитів. Пауза на 5 секунд...")
                        await asyncio.sleep(5) 
                except Exception as e:
                    print(f"  Загальна помилка вибірки {query}: {e}")
        
        # Deduplicate by coordinates
        unique_addresses = []
        seen = set()
        for addr in addresses:
            key = (round(addr['lat'], 5), round(addr['lon'], 5)) 
            if key not in seen:
                seen.add(key)
                unique_addresses.append(addr)
        
        print(f"Усього унікальних адрес у кеші: {len(unique_addresses)}")
        return unique_addresses
    
    def _get_zone(self, lat: float, lon: float) -> str:
        """Assign zone based on location (simple grid-based approximation)"""
        lat_range = self.city.bounds['max_lat'] - self.city.bounds['min_lat']
        
        # Обчислюємо індекс, до якого 'тягнеться' координата, відносно кількості зон
        lat_idx = int((lat - self.city.bounds['min_lat']) / lat_range * len(self.city.zones))
        
        # Забезпечуємо, що індекс знаходиться в межах списку зон
        lat_idx = max(0, min(lat_idx, len(self.city.zones) - 1))
        return self.city.zones[lat_idx]
    
    def get_random_address(self) -> Dict:
        """Get random address"""
        if not self.addresses:
            raise ValueError(f"No addresses available for {self.city.name}")
        return random.choice(self.addresses)
    
    def get_random_pair(self) -> tuple[Dict, Dict]:
        """Get random pickup and delivery addresses"""
        pickup = self.get_random_address()
        delivery = self.get_random_address()
        
        # Ensure they're different
        while delivery == pickup:
            delivery = self.get_random_address()
        
        return pickup, delivery
    
    def get_by_zone(self, zone: str) -> List[Dict]:
        """Get all addresses in a specific zone"""
        return [addr for addr in self.addresses if addr['zone'] == zone]
