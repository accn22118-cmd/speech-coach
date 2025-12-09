"""
Модуль для управления жизненным циклом приложения.
Обрабатывает запуск и завершение ресурсов.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from app.api.deps import get_gigachat_client

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Контекстный менеджер для управления жизненным циклом приложения.
    """
    # Запуск приложения
    logger.info("Starting Speech Coach API")

    # Предварительная инициализация GigaChat клиента
    # для проверки подключения на старте
    gigachat_client = get_gigachat_client()
    if gigachat_client:
        logger.info("GigaChat client initialized and ready")
    else:
        logger.info("GigaChat client not available")

    yield

    # Завершение работы приложения
    logger.info("Shutting down Speech Coach API")

    # Закрываем GigaChat клиент, если он был создан
    if gigachat_client:
        try:
            await gigachat_client.close()
            logger.info("GigaChat client closed successfully")
        except Exception as e:
            logger.error(f"Error closing GigaChat client: {e}")
