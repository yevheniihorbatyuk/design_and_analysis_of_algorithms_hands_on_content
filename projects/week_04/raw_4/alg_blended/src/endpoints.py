from fastapi import APIRouter, HTTPException, Depends, Request, Query
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
from enum import Enum
import json

from .caches import BaseCache
from .rate_limit import BaseRateLimiter
from .base import BaseExternalAPI
from .providers import (
    WeatherProvider, 
    ZippoProvider,
    CurrencyProvider,
    NewsProvider,
    TranslationProvider
)
from config import get_settings, get_cache, get_rate_limiter

router = APIRouter()
settings = get_settings()

# Initialize providers
providers = {
    "weather": WeatherProvider(),
    "zippo": ZippoProvider(),
    "currency": CurrencyProvider(),
    "news": NewsProvider(),
    "translation": TranslationProvider()
}

class NewsCategory(str, Enum):
    GENERAL = "general"
    BUSINESS = "business"
    TECHNOLOGY = "technology"
    SPORTS = "sports"
    ENTERTAINMENT = "entertainment"
    SCIENCE = "science"
    HEALTH = "health"

async def check_rate_limit(request: Request) -> None:
    """
    Check if the request is allowed by the rate limiter
    """
    client_ip = request.client.host
    
    # Get rate limiter directly instead of as a dependency
    rate_limiter = get_rate_limiter()

    
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later."
        )
    #     print('429 : Too many requests. Please try again later.')

async def get_cached_response(
    cache_key: str,
    provider: BaseExternalAPI,
    params: Dict[str, Any],
    cache: BaseCache = Depends(get_cache)
) -> Dict[str, Any]:
    cached_result = cache.get(cache_key)
    if cached_result:
        return {"source": "cache", "results": cached_result}

    try:
        result = await provider.execute_request(params)
        cache.put(cache_key, result)
        return {"source": "provider", "results": result}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Provider error: {str(e)}"
        )

@router.get('/weather/{city}')
async def get_weather(
    request: Request,
    city: str,
    detailed: bool = Query(False, description="Get detailed weather information"),
    cache: BaseCache = Depends(get_cache),
):
    await check_rate_limit(request)
    cache_key = f"weather:{city}:{'detailed' if detailed else 'basic'}"
    return await get_cached_response(
        cache_key,
        providers["weather"],
        {"destination": city, "detailed": detailed},
        cache
    )

@router.get('/zipcode/{country}/{code}')
async def get_zipcode(
    request: Request,
    country: str,
    code: str,
    cache: BaseCache = Depends(get_cache),
):
    await check_rate_limit(request)  # No Depends() here
    cache_key = f"zipcode:{country}:{code}"
    return await get_cached_response(
        cache_key,
        providers["zippo"],
        {"country": country, "zipcode": code},
        cache
    )

@router.get('/currency/convert')
async def convert_currency(
    request: Request,
    base: str = Query(..., description="Base currency code (e.g., USD)"),
    targets: List[str] = Query(..., description="Target currency codes"),
    amount: float = Query(1.0, description="Amount to convert"),
    cache: BaseCache = Depends(get_cache),
):
    await check_rate_limit(request)
    cache_key = f"currency:{base}:{'-'.join(sorted(targets))}:{amount}"
    return await get_cached_response(
        cache_key,
        providers["currency"],
        {"base": base, "targets": targets, "amount": amount},
        cache
    )

@router.get('/news/search')
async def search_news(
    request: Request,
    query: Optional[str] = None,
    category: NewsCategory = NewsCategory.GENERAL,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    cache: BaseCache = Depends(get_cache),
):
    await check_rate_limit(request)
    cache_key = f"news:{query}:{category}:{page}:{page_size}"
    return await get_cached_response(
        cache_key,
        providers["news"],
        {
            "query": query,
            "category": category,
            "page": page,
            "page_size": page_size
        },
        cache
    )

@router.get('/translate')
async def translate_text(
    request: Request,
    text: str = Query(..., min_length=1, max_length=5000),
    source_lang: str = Query("auto", min_length=2, max_length=2),
    target_lang: str = Query(..., min_length=2, max_length=2),
    cache: BaseCache = Depends(get_cache),
):
    await check_rate_limit(request)
    cache_key = f"translate:{source_lang}:{target_lang}:{hash(text)}"
    return await get_cached_response(
        cache_key,
        providers["translation"],
        {
            "text": text,
            "source": source_lang,
            "target": target_lang
        },
        cache
    )

@router.get('/multi/{provider}/{action}')
async def multi_provider_request(
    request: Request,
    provider: str,
    action: str,
    params_json: str = Query(...),
    cache: BaseCache = Depends(get_cache),
):
    try:
        params = json.loads(params_json)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON in params_json query parameter"
        )

    await check_rate_limit(request)
    cache_key = f"{provider}:{action}:{hash(str(sorted(params.items())))}"
    return await get_cached_response(
        cache_key,
        providers[provider],
        {"action": action, **params},
        cache
    )

# @router.get('/batch')
# async def batch_request(
#     request: Request,
#     requests: List[Dict[str, Any]] = Query(...),
#     cache: BaseCache = Depends(get_cache),
# ):
#     await check_rate_limit(request)
    
#     results = []
#     for req in requests:
#         provider_name = req.get("provider")
#         if provider_name not in providers:
#             results.append({
#                 "error": f"Provider '{provider_name}' not found"
#             })
#             continue

#         cache_key = f"batch:{provider_name}:{hash(str(sorted(req.items())))}"
#         try:
#             result = await get_cached_response(
#                 cache_key,
#                 providers[provider_name],
#                 req.get("params", {}),
#                 cache
#             )
#             results.append(result)
#         except Exception as e:
#             results.append({
#                 "error": str(e)
#             })

#     return {"results": results}

@router.get('/health')
async def health_check():
    provider_status = {}
    for name, provider in providers.items():
        try:
            await provider.health_check()
            provider_status[name] = "healthy"
        except Exception as e:
            provider_status[name] = f"unhealthy: {str(e)}"

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "providers": provider_status
    }