import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.health import router as health_router
from app.api.routes.analysis import router as analysis_router
from app.core.lifespan import lifespan
from app.core.config import settings

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Speech Coach API",
    description="Сервис для анализа качества публичной речи с поддержкой AI-анализа через GigaChat",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware - настройте под свои нужды
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене укажите конкретные origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Включаем роутеры
app.include_router(health_router)
app.include_router(analysis_router)


@app.get("/")
async def root():
    """Корневой эндпоинт с информацией о API"""
    return {
        "name": "Speech Coach API",
        "version": "1.0.0",
        "description": "Анализ качества публичной речи с AI-анализом",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "analyze": "/api/v1/analyze"
        },
        "features": {
            "whisper_transcription": True,
            "speech_metrics": True,
            "gigachat_analysis": settings.gigachat_enabled
        }
    }
