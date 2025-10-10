# services/generator/generators.py

import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import uuid


@dataclass
class Order:
    order_id: str
    customer_id: str
    city: str
    pickup_lat: float
    pickup_lon: float
    pickup_address: str
    pickup_zone: str
    delivery_lat: float
    delivery_lon: float
    delivery_address: str
    delivery_zone: str
    weight: float
    priority: str
    time_window_start: str
    time_window_end: str
    timestamp: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class GPSEvent:
    vehicle_id: str
    driver_id: str
    city: str
    lat: float
    lon: float
    speed: float
    heading: float
    status: str
    current_load: float
    timestamp: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class VehicleStatus:
    vehicle_id: str
    driver_id: str
    city: str
    status: str
    capacity: float
    current_load: float
    orders_completed_today: int
    timestamp: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


class OrderGenerator:
    """Generates realistic orders"""
    
    def __init__(self, address_provider, city_name: str):
        self.address_provider = address_provider
        self.city_name = city_name
        self.customer_pool = [f"CUST-{i:06d}" for i in range(1, 10001)]
    
    def generate(self) -> Order:
        """Generate single order"""
        pickup, delivery = self.address_provider.get_random_pair()
        
        now = datetime.now()
        time_window_start = now + timedelta(hours=random.randint(1, 4))
        time_window_end = time_window_start + timedelta(hours=2)
        
        return Order(
            order_id=f"ORD-{uuid.uuid4().hex[:12]}",
            customer_id=random.choice(self.customer_pool),
            city=self.city_name,
            pickup_lat=pickup['lat'],
            pickup_lon=pickup['lon'],
            pickup_address=pickup['display_name'],
            pickup_zone=pickup['zone'],
            delivery_lat=delivery['lat'],
            delivery_lon=delivery['lon'],
            delivery_address=delivery['display_name'],
            delivery_zone=delivery['zone'],
            weight=round(random.uniform(1.0, 50.0), 2),
            priority=random.choices(['standard', 'urgent', 'express'], 
                                   weights=[0.7, 0.2, 0.1])[0],
            time_window_start=time_window_start.isoformat(),
            time_window_end=time_window_end.isoformat(),
            timestamp=now.isoformat()
        )


class VehicleSimulator:
    """Simulates vehicle movement and status"""
    
    def __init__(self, vehicle_id: str, driver_id: str, city_name: str, 
                 address_provider, capacity: float = 100.0):
        self.vehicle_id = vehicle_id
        self.driver_id = driver_id
        self.city_name = city_name
        self.address_provider = address_provider
        self.capacity = capacity
        
        # Initial state
        start_addr = address_provider.get_random_address()
        self.lat = start_addr['lat']
        self.lon = start_addr['lon']
        self.speed = 0.0
        self.heading = random.uniform(0, 360)
        self.status = 'available'
        self.current_load = 0.0
        self.orders_completed = 0
        
        # Route simulation
        self.destination = None
        self.route_progress = 0.0
    
    def update(self) -> tuple[GPSEvent, Optional[VehicleStatus]]:
        """Update vehicle state and return events"""
        now = datetime.now()
        
        # Simulate movement
        if self.status == 'busy' and self.destination:
            # Move towards destination
            self._move_towards_destination()
            
            # Check if reached
            if self._distance_to_destination() < 0.001:  # ~100m
                self._handle_arrival()
        
        elif self.status == 'available':
            # Random idle movement
            self.speed = random.uniform(0, 5)
            self._random_move()
        
        # Generate GPS event
        gps_event = GPSEvent(
            vehicle_id=self.vehicle_id,
            driver_id=self.driver_id,
            city=self.city_name,
            lat=round(self.lat, 6),
            lon=round(self.lon, 6),
            speed=round(self.speed, 2),
            heading=round(self.heading, 2),
            status=self.status,
            current_load=round(self.current_load, 2),
            timestamp=now.isoformat()
        )
        
        # Occasionally generate status update
        status_event = None
        if random.random() < 0.1:  # 10% chance
            status_event = VehicleStatus(
                vehicle_id=self.vehicle_id,
                driver_id=self.driver_id,
                city=self.city_name,
                status=self.status,
                capacity=self.capacity,
                current_load=round(self.current_load, 2),
                orders_completed_today=self.orders_completed,
                timestamp=now.isoformat()
            )
        
        return gps_event, status_event
    
    def assign_order(self, order: Order):
        """Assign order to vehicle"""
        self.status = 'busy'
        self.destination = {
            'lat': order.delivery_lat,
            'lon': order.delivery_lon
        }
        self.current_load += order.weight
        self.route_progress = 0.0
    
    def _move_towards_destination(self):
        """Move vehicle towards destination"""
        if not self.destination:
            return
        
        # Simple linear interpolation
        dlat = self.destination['lat'] - self.lat
        dlon = self.destination['lon'] - self.lon
        
        # Speed: 30-60 km/h
        self.speed = random.uniform(30, 60)
        
        # Update heading
        import math
        self.heading = math.degrees(math.atan2(dlon, dlat)) % 360
        
        # Move step (~100m per update at 40km/h with 10s interval)
        step = 0.0009  # ~100m in degrees
        self.lat += dlat * step / abs(dlat + 0.0001)
        self.lon += dlon * step / abs(dlon + 0.0001)
        
        self.route_progress += step
    
    def _distance_to_destination(self) -> float:
        """Calculate distance to destination"""
        if not self.destination:
            return float('inf')
        
        dlat = abs(self.destination['lat'] - self.lat)
        dlon = abs(self.destination['lon'] - self.lon)
        return (dlat ** 2 + dlon ** 2) ** 0.5
    
    def _random_move(self):
        """Random idle movement"""
        self.lat += random.uniform(-0.0005, 0.0005)
        self.lon += random.uniform(-0.0005, 0.0005)
        self.heading = (self.heading + random.uniform(-30, 30)) % 360
    
    def _handle_arrival(self):
        """Handle arrival at destination"""
        self.status = 'available'
        self.destination = None
        self.current_load = max(0, self.current_load - random.uniform(5, 20))
        self.orders_completed += 1
        self.speed = 0