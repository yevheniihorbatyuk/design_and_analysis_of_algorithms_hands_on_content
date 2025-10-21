from typing import Dict, Any
import asyncio
from openmeteo_requests import Client
import requests_cache
from retry_requests import retry
from datetime import datetime
from .base import BaseExternalAPI
import aiohttp

class WeatherProvider(BaseExternalAPI):
    def __init__(self):
        super().__init__("WeatherProvider", rate_limit=10, time_window=1)
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.client = Client(session=retry_session)

    async def search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        coordinates = {
            "Paris": {"latitude": 48.8566, "longitude": 2.3522},
            "London": {"latitude": 51.5074, "longitude": -0.1278},
            "New York": {"latitude": 40.7128, "longitude": -74.0060}
        }
        
        location = params.get("destination", "Paris")
        coords = coordinates.get(location, coordinates["Paris"])
        
        params = {
            "latitude": coords["latitude"],
            "longitude": coords["longitude"],
            "hourly": ["temperature_2m", "precipitation_probability"],
            "timezone": "auto"
        }
        
        responses = self.client.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
        response = responses[0]
        
        hourly = response.Hourly()
        time = [datetime.fromisoformat(t).strftime("%Y-%m-%d %H:%M") 
               for t in hourly.Time()]
        temp = hourly.Variables(0).ValuesAsNumpy()
        precip = hourly.Variables(1).ValuesAsNumpy()

        forecast = []
        for i in range(len(time)):
            forecast.append({
                "time": time[i],
                "temperature": float(temp[i]),
                "precipitation_probability": float(precip[i])
            })

        return {
            "provider": self.name,
            "location": location,
            "coordinates": coords,
            "forecast": forecast[:24]  # Return first 24 hours
        }



class ZippoProvider(BaseExternalAPI):
   def __init__(self):
       super().__init__("ZippoProvider", rate_limit=10, time_window=1)
       self.base_url = "https://api.zippopotam.us"

   async def search(self, params: Dict[str, Any]) -> Dict[str, Any]:
       import requests
       zipcode = params.get("zipcode")
       country = params.get("country", "us")
       
       if not zipcode:
           raise ValueError("Zipcode is required")

       response = requests.get(f"{self.base_url}/{country}/{zipcode}")
       
       if response.status_code == 404:
           return {
               "provider": self.name,
               "found": False,
               "message": "ZIP code not found"
           }
       
       data = response.json()
       return {
           "provider": self.name,
           "found": True, 
           "country": data.get("country"),
           "country_abbreviation": data.get("country abbreviation"),
           "places": data.get("places", [])
       }

class CurrencyProvider(BaseExternalAPI):
    def __init__(self):
        super().__init__("CurrencyProvider", rate_limit=30, time_window=60)
        self.base_url = "https://api.exchangerate-api.com/v4/latest"

    async def search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        base_currency = params.get("base", "USD")
        target_currencies = params.get("targets", ["EUR", "GBP", "JPY"])
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/{base_currency}") as response:
                data = await response.json()
                
                results = {
                    "provider": self.name,
                    "base": base_currency,
                    "timestamp": data.get("time_last_updated"),
                    "rates": {
                        currency: data["rates"][currency]
                        for currency in target_currencies
                        if currency in data["rates"]
                    }
                }
                return results

class NewsProvider(BaseExternalAPI):
    def __init__(self):
        super().__init__("NewsProvider", rate_limit=100, time_window=60)
        self.api_key = "your_api_key"
        self.base_url = "https://newsapi.org/v2"

    async def search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("query", "")
        category = params.get("category", "general")
        
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/top-headlines"
            params = {
                "q": query,
                "category": category,
                "apiKey": self.api_key
            }
            
            async with session.get(url, params=params) as response:
                data = await response.json()
                return {
                    "provider": self.name,
                    "query": query,
                    "category": category,
                    "articles": data.get("articles", [])
                }

class TranslationProvider(BaseExternalAPI):
    def __init__(self):
        super().__init__("TranslationProvider", rate_limit=60, time_window=60)
        self.api_key = "your_api_key"
        self.base_url = "https://translation-api.com/v1"

    async def search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        text = params.get("text", "")
        source_lang = params.get("source", "auto")
        target_lang = params.get("target", "en")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/translate",
                params={
                    "text": text,
                    "source": source_lang,
                    "target": target_lang,
                    "key": self.api_key
                }
            ) as response:
                data = await response.json()
                return {
                    "provider": self.name,
                    "original": text,
                    "translated": data.get("translated_text"),
                    "source_language": data.get("detected_source_language", source_lang),
                    "target_language": target_lang
                }