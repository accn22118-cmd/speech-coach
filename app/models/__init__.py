# Экспортируем все модели для удобного импорта
<<<<<<< HEAD
from .transcript import Transcript, TranscriptSegment, WordTiming
from .timed_analysis import (
    TimedAnalysisResult, TimedFillerWord, TimedPause,
    SpeechRateWindow, EmotionalPeak, AdviceItem,
    FillerWordsStats, PausesStats, PhraseStats
)
=======
from app.models.transcript import Transcript, TranscriptSegment
from app.models.analysis import (
    AnalysisResult,
    FillerWordsStats,
    PausesStats,
    PhraseStats,
    AdviceItem
)
from app.models.gigachat import GigaChatAnalysis
>>>>>>> feature/gigachat-integration

__all__ = [
    "Transcript",
    "TranscriptSegment",
<<<<<<< HEAD
    "WordTiming",
    "TimedAnalysisResult",
    "TimedFillerWord",
    "TimedPause",
    "SpeechRateWindow",
    "EmotionalPeak",
    "AdviceItem",
    "FillerWordsStats",
    "PausesStats",
    "PhraseStats",
=======
    "AnalysisResult",
    "FillerWordsStats",
    "PausesStats",
    "PhraseStats",
    "AdviceItem",
    "GigaChatAnalysis",
>>>>>>> feature/gigachat-integration
]
