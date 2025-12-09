import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional
import logging

from fastapi import UploadFile

from app.services.audio_extractor import AudioExtractor
from app.services.transcriber import Transcriber
from app.services.analyzer import SpeechAnalyzer
from app.services.gigachat import GigaChatClient
from app.models.analysis import AnalysisResult

logger = logging.getLogger(__name__)


class SpeechAnalysisPipeline:
    """
    Координирует:
    - приём UploadFile (видео),
    - сохранение во временный файл,
    - извлечение аудио,
    - транскрибацию,
    - анализ,
    - расширенный анализ через GigaChat (если включен).
    """

    def __init__(
        self,
        audio_extractor: AudioExtractor,
        transcriber: Transcriber,
        analyzer: SpeechAnalyzer,
        gigachat_client: Optional[GigaChatClient] = None,
    ):
        self.audio_extractor = audio_extractor
        self.transcriber = transcriber
        self.analyzer = analyzer
        self.gigachat_client = gigachat_client

    async def analyze_upload(self, file: UploadFile) -> AnalysisResult:
        """
        Анализирует загруженное видео и возвращает результаты.
        Включает расширенный анализ через GigaChat, если настроен.
        """
        suffix = Path(file.filename or "video").suffix or ".mp4"

        tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_video_path = Path(tmp_video.name)
        tmp_video.close()

        await self._save_upload_to_path(file, temp_video_path)

        temp_audio_path = temp_video_path.with_suffix(".wav")

        try:
            # 1) Извлекаем аудио из видео
            logger.info(f"Extracting audio from {temp_video_path}")
            self.audio_extractor.extract(temp_video_path, temp_audio_path)

            # 2) Транскрибируем аудио
            logger.info("Transcribing audio...")
            transcript = self.transcriber.transcribe(temp_audio_path)

            # 3) Анализируем с учётом пути к аудио (для оценки пауз)
            logger.info("Analyzing speech metrics...")
            result = self.analyzer.analyze(
                transcript, audio_path=temp_audio_path)

            # 4) Запрашиваем расширенный анализ через GigaChat, если настроен
            if self.gigachat_client:
                logger.info("Requesting GigaChat analysis...")
                try:
                    gigachat_analysis = await self.gigachat_client.analyze_speech(result)
                    if gigachat_analysis:
                        result.gigachat_analysis = gigachat_analysis
                        logger.info("GigaChat analysis completed successfully")
                    else:
                        logger.warning("GigaChat analysis returned no results")
                except Exception as e:
                    logger.error(f"GigaChat analysis failed: {e}")
                    # Не прерываем основной анализ из-за ошибки GigaChat

            logger.info("Analysis completed successfully")
            return result

        finally:
            # Удаляем временные файлы
            for path in (temp_video_path, temp_audio_path):
                try:
                    if path.exists():
                        os.remove(path)
                except OSError as e:
                    logger.warning(f"Failed to delete temp file {path}: {e}")

    @staticmethod
    async def _save_upload_to_path(upload: UploadFile, dst: Path) -> None:
        upload.file.seek(0)
        with dst.open("wb") as out_file:
            shutil.copyfileobj(upload.file, out_file)
        await upload.close()
