from fastapi import FastAPI
from src.endpoints import router
from config import get_settings
import logging
import uvicorn

log_config = uvicorn.config.LOGGING_CONFIG
log_config["formatters"]["default"]["fmt"] = "%(levelname)s: %(name)s: %(message)s"
log_config["formatters"]["access"]["fmt"] = '%(levelname)s: %(name)s: %(client_addr)s - "%(request_line)s" %(status_code)s'
log_config["loggers"]["uvicorn"]["level"] = "DEBUG"
log_config["loggers"]["uvicorn.access"]["level"] = "DEBUG"
log_config["loggers"]["uvicorn.error"]["level"] = "DEBUG"

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    description="Service Aggregator with Rate Limiting and Caching"
)

app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8899, log_config=log_config)