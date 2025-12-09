import logging
from functools import lru_cache
from typing import Optional

from app.services.audio_extractor import FfmpegAudioExtractor
from app.services.transcriber import LocalWhisperTranscriber
from app.services.analyzer import SpeechAnalyzer
from app.services.gigachat import GigaChatClient
from app.services.pipeline import SpeechAnalysisPipeline
from app.core.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_audio_extractor() -> FfmpegAudioExtractor:
    """Создает и кеширует экстрактор аудио через FFmpeg"""
    return FfmpegAudioExtractor()


@lru_cache(maxsize=1)
def get_transcriber() -> LocalWhisperTranscriber:
    """Создает и кеширует трансбайбер Whisper"""
    return LocalWhisperTranscriber()


@lru_cache(maxsize=1)
def get_analyzer() -> SpeechAnalyzer:
    """Создает и кеширует анализатор речи"""
    return SpeechAnalyzer()


@lru_cache(maxsize=1)
def get_gigachat_client() -> Optional[GigaChatClient]:
    """Создает и кеширует клиент GigaChat, если настроен и включен"""
    if not settings.gigachat_enabled:
        logger.info("GigaChat is disabled in settings")
        return None

    if not settings.gigachat_api_key:
        logger.warning("GigaChat API key not configured")
        return None

    try:
        client = GigaChatClient()
        logger.info("GigaChat client initialized successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize GigaChat client: {e}")
        return None


@lru_cache(maxsize=1)
def get_speech_pipeline() -> SpeechAnalysisPipeline:
    """
    Создаёт и кеширует единственный экземпляр пайплайна на процесс.
    Включает GigaChat клиент, если настроен и включен.
    """
    extractor = get_audio_extractor()
    transcriber = get_transcriber()
    analyzer = get_analyzer()
    gigachat_client = get_gigachat_client()

    logger.info(f"Initializing speech pipeline (GigaChat: {
                'enabled' if gigachat_client else 'disabled'})")

    return SpeechAnalysisPipeline(
        audio_extractor=extractor,
        transcriber=transcriber,
        analyzer=analyzer,
        gigachat_client=gigachat_client,
    )
