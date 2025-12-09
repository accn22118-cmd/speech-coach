import logging
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status

from app.api.deps import get_speech_pipeline
<<<<<<< HEAD
from app.models.timed_analysis import TimedAnalysisResult
from app.services.pipeline import SpeechAnalysisPipeline
=======
from app.models.analysis import AnalysisResult
from app.core.exceptions import (
    FileValidationError,
    TranscriptionError,
    AnalysisError,
)
>>>>>>> feature/gigachat-integration

router = APIRouter(prefix="/api/v1", tags=["analysis"])
logger = logging.getLogger(__name__)


<<<<<<< HEAD
@router.post("/analyze", response_model=TimedAnalysisResult)
async def analyze_video(
    file: UploadFile = File(...),
    pipeline: SpeechAnalysisPipeline = Depends(get_speech_pipeline),
):
    """
    Анализ речи из видео с временными метками.

    Возвращает детализированные данные с таймингами для:
    - каждого слова-паразита
    - каждой паузы с контекстом
    - темпа речи в разных временных окнах
    - активности говорения
    - границ фраз
    - сегментов транскрипта с таймингами слов
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename")

    try:
        result = await pipeline.analyze(file)
=======
@router.post(
    "/analyze",
    response_model=AnalysisResult,
    summary="Анализ видеофайла с речью",
    description="""
    Анализирует видеофайл и возвращает метрики качества речи.
    
    Поддерживаемые форматы: MP4, MOV, AVI, MKV, WEBM, FLV, WMV, M4V
    Максимальный размер файла: 100 MB
    """,
    responses={
        200: {"description": "Анализ успешно выполнен"},
        400: {"description": "Некорректный файл или формат"},
        413: {"description": "Файл слишком большой"},
        500: {"description": "Ошибка при обработке файла"},
    }
)
async def analyze_video(
    file: UploadFile = File(...,
                            description="Видеофайл для анализа (до 100 MB)"),
    pipeline=Depends(get_speech_pipeline),
) -> AnalysisResult:
    """
    Анализирует загруженное видео и возвращает результаты анализа речи.
    """
    logger.info(f"Получен запрос на анализ файла: {file.filename}")

    try:
        result = await pipeline.analyze_upload(file)
        logger.info(f"Анализ завершен для {file.filename}")
>>>>>>> feature/gigachat-integration
        return result

    except FileValidationError as e:
        logger.warning(f"Ошибка валидации файла {file.filename}: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except (TranscriptionError, AnalysisError) as e:
        logger.error(f"Ошибка обработки {file.filename}: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except Exception as e:
        logger.error(f"Неожиданная ошибка для {
                     file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Внутренняя ошибка сервера при обработке файла"
        )
