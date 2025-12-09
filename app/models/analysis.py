from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel, Field


class FillerWordsStats(BaseModel):
    total: int
    per_100_words: float
    items: List[Dict[str, Any]]  # {"word": str, "count": int}


class PausesStats(BaseModel):
    count: int
    avg_sec: float
    max_sec: float
    long_pauses: List[Dict[str, float]]


class PhraseStats(BaseModel):
    count: int
    avg_words: float
    avg_duration_sec: float
    min_words: int
    max_words: int
    min_duration_sec: float
    max_duration_sec: float
    length_classification: str
    rhythm_variation: str


class AdviceItem(BaseModel):
    category: Literal["speech_rate", "filler_words", "pauses", "phrasing"]
    severity: Literal["info", "suggestion", "warning"]
    title: str
    observation: str
    recommendation: str


class AnalysisResult(BaseModel):
    duration_sec: float
    speaking_time_sec: float
    speaking_ratio: float
    words_total: int
    words_per_minute: float
    filler_words: FillerWordsStats
    pauses: PausesStats
    phrases: PhraseStats
    advice: List[AdviceItem]
    transcript: str
    gigachat_analysis: Optional[Any] = None
