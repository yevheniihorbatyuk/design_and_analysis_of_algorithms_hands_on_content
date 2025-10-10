# services/generator/config.py

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import yaml


@dataclass
class CityConfig:
    """Configuration for a city"""
    name: str
    country: str
    bounds: dict  # {'min_lat': ..., 'max_lat': ..., 'min_lon': ..., 'max_lon': ...}
    center: dict  # {'lat': ..., 'lon': ...}
    zones: List[str]  # ['Downtown', 'Airport', 'Suburbs', ...]
    region: Optional[str] = None  

        # Demand modeling
    population: int = 1000000
    area_km2: float = 100.0
    gdp_per_capita: float = 5000.0
    delivery_culture: float = 1.0
    urbanization: float = 1.0
    zone_weights: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """Calculate derived metrics"""
        if self.zone_weights is None:
            self.zone_weights = {zone: 1.0 for zone in self.zones}
        
        # Calculate demand multiplier
        gdp_factor = min(2.0, max(0.5, self.gdp_per_capita / 5000))
        urban_factor = 0.7 + (self.urbanization * 0.8)
        
        self.demand_multiplier = gdp_factor * urban_factor * self.delivery_culture
        
        # Daily orders
        self.base_orders_per_day = 500
        self.estimated_daily_orders = int(
            (self.population / 1_000_000) * 
            self.base_orders_per_day * 
            self.demand_multiplier
        )
        
        self.density = self.population / self.area_km2
    
    def get_zone_probability(self, zone: str) -> float:
        """Get probability of order in this zone"""
        total_weight = sum(self.zone_weights.values())
        return self.zone_weights.get(zone, 1.0) / total_weight
    
    
@dataclass
class GeneratorConfig:
    """Main generator configuration"""
    cities: List[CityConfig]
    total_orders_per_minute: int = 50
    auto_distribute_orders: bool = True
    vehicles_per_1000_orders: int = 10
    gps_interval_seconds: int = 10
    peak_hours: List[int] = field(default_factory=lambda: [8, 9, 17, 18, 19])
    peak_multiplier: float = 2.0
    weekend_multiplier: float = 0.7
    night_multiplier: float = 0.3
    output_mode: str = "kafka"
    kafka_broker: str = "localhost:9092"
    output_dir: str = "./data/generated"
    
    def __post_init__(self):
        """Calculate order distribution"""
        if self.auto_distribute_orders and len(self.cities) > 1:
            total_demand = sum(city.estimated_daily_orders for city in self.cities)
            
            for city in self.cities:
                city.order_share = city.estimated_daily_orders / total_demand
                city.orders_per_minute = int(
                    self.total_orders_per_minute * city.order_share
                )
                city.orders_per_minute = max(1, city.orders_per_minute)
                daily_orders = city.orders_per_minute * 60 * 24
                city.num_vehicles = max(5, int(
                    daily_orders / 1000 * self.vehicles_per_1000_orders
                ))
        else:
            per_city = self.total_orders_per_minute // len(self.cities)
            for city in self.cities:
                city.order_share = 1.0 / len(self.cities)
                city.orders_per_minute = max(1, per_city)
                city.num_vehicles = max(5, per_city * 24 * 60 // 100)
    
    @classmethod
    def from_yaml(cls, path: str):
        """Load config from YAML"""
        with open(path) as f:
            data = yaml.safe_load(f)
        cities = [CityConfig(**city) for city in data['cities']]
        config_dict = {k: v for k, v in data.items() if k != 'cities'}
        config_dict['cities'] = cities
        return cls(**config_dict)


# Predefined city configurations
CITY_CONFIGS = {
    'kyiv': CityConfig(
        name='Kyiv',
        country='Ukraine',
        bounds={
            'min_lat': 50.213, 'max_lat': 50.590,
            'min_lon': 30.239, 'max_lon': 30.825
        },
        center={'lat': 50.4501, 'lon': 30.5234},
        zones=['Pechersk', 'Shevchenko', 'Podil', 'Obolon', 
               'Darnytsia', 'Desnyan', 'Holosiiv', 'Solomyan']
    ),
    'lviv': CityConfig(
        name='Lviv',
        country='Ukraine',
        bounds={
            'min_lat': 49.770, 'max_lat': 49.905,
            'min_lon': 23.930, 'max_lon': 24.145
        },
        center={'lat': 49.8397, 'lon': 24.0297},
        zones=['Center', 'Lychakiv', 'Sykhiv', 'Frankivskyi']
    ),
    'berlin': CityConfig(
        name='Berlin',
        country='Germany',
        bounds={
            'min_lat': 52.338, 'max_lat': 52.675,
            'min_lon': 13.088, 'max_lon': 13.761
        },
        center={'lat': 52.5200, 'lon': 13.4050},
        zones=['Mitte', 'Charlottenburg', 'Kreuzberg', 'Prenzlauer Berg']
    ),
    'paris': CityConfig(
        name='Paris',
        country='France',
        bounds={
            'min_lat': 48.815, 'max_lat': 48.902,
            'min_lon': 2.225, 'max_lon': 2.470
        },
        center={'lat': 48.8566, 'lon': 2.3522},
        zones=['1st', '2nd', '8th', '16th', 'Montmartre']
    )
}