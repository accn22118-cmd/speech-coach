from typing import List, Dict, Tuple, Any
import re
import wave
import struct
import math
from pathlib import Path

from app.models.transcript import Transcript, TranscriptSegment, WordTiming
from app.models.timed_analysis import (
    TimedAnalysisResult, TimedFillerWord, TimedPause, SpeechRateWindow,
    EmotionalPeak, AdviceItem, FillerWordsStats, PausesStats, PhraseStats
)

# --------------------
# Константы "нормы"
# --------------------

MIN_PAUSE_GAP_SEC = 0.5
MIN_COMFORT_WPM = 100.0
MAX_COMFORT_WPM = 180.0
LONG_PAUSE_SEC = 2.5
SILENCE_FACTOR = 0.35
PAUSE_SEGMENT_TIME_TOLERANCE = 0.25

# --------------------
# Слова-паразиты
# --------------------

FILLER_DEFINITIONS: List[Tuple[str, str]] = [
<<<<<<< HEAD
    # Русские
    ("э-э", r"\bэ+([- ]э+)*\b"),
    ("ну", r"\bну\b"),
    ("вот", r"\bвот\b"),
=======
    # Русские звучания
    ("э-э", r"\bэ+([- ]э+)*\b"),
    ("ну", r"\bну\b"),
    ("вот", r"\bвот\b"),

    # Русские конструкции
>>>>>>> feature/gigachat-integration
    ("как бы", r"\bкак бы\b"),
    ("типа", r"\bтипа\b"),
    ("то есть", r"\bто есть\b"),
    ("значит", r"\bзначит\b"),
    ("получается", r"\bполучается\b"),

<<<<<<< HEAD
    # Английские
    ("uh", r"\buh+\b"),
    ("um", r"\bum+\b"),
    ("er", r"\ber+\b"),
    ("ah", r"\bah+\b"),
=======
    # Английские звучания
    ("uh", r"\buh+\b"),
    ("um", r"\bum+\b"),
    ("er", r"\ber+\b"),

    # Английские конструкции
>>>>>>> feature/gigachat-integration
    ("like", r"\blike\b"),
    ("so", r"\bso\b"),
    ("you know", r"\byou know\b"),
    ("i mean", r"\bi mean\b"),
]

COMPILED_FILLERS: List[Tuple[str, re.Pattern]] = [
    (name, re.compile(pattern, flags=re.IGNORECASE | re.MULTILINE))
    for name, pattern in FILLER_DEFINITIONS
]


class SpeechAnalyzer:
<<<<<<< HEAD
    """
    Анализирует транскрипт и формирует метрики/советы с временными метками.
    """

=======
>>>>>>> feature/gigachat-integration
    def analyze(
        self,
        transcript: Transcript,
        audio_path: Path | None = None,
<<<<<<< HEAD
    ) -> TimedAnalysisResult:
        """
        Основной метод анализа, возвращает полный результат с таймингами.
        """
        # Базовые расчеты
        duration_sec, speaking_time_sec, words_total = self._calculate_basic_metrics(
            transcript)
        words_per_minute = self._calculate_wpm(words_total, speaking_time_sec)
        speaking_ratio = speaking_time_sec / duration_sec if duration_sec > 0 else 0.0

        # Анализ слов-паразитов
        filler_total, filler_detail = self._count_fillers(transcript.text)
        filler_stats = self._build_filler_stats(
            filler_total, filler_detail, words_total)
        filler_detailed = self._find_fillers_with_timings(transcript)

        # Анализ пауз
        raw_pauses = self._extract_raw_pauses(transcript.segments)
        if audio_path and raw_pauses:
            filtered_pauses = self._filter_noisy_pauses(
                audio_path, raw_pauses, transcript.segments)
        else:
            filtered_pauses = raw_pauses

        pauses_stats = self._summarize_pauses(filtered_pauses)
        pauses_detailed = self._analyze_pauses_with_context(
            filtered_pauses, transcript.segments)

        # Анализ фраз
        phrase_stats = self._build_phrase_stats(
            transcript.segments, filtered_pauses)

        # Генерация советов
=======
    ) -> AnalysisResult:
        segments = transcript.segments
        if not segments:
            return self._empty_analysis_result(transcript.text or "")

        duration_sec = float(segments[-1].end)
        full_text = transcript.text or ""
        words = self._split_words(full_text)
        words_total = len(words)

        # Расчет времени говорения и пауз
        speaking_time_sec, pauses_raw = self._calculate_speaking_stats(
            segments)

        # Фильтрация пауз по аудио
        pauses_filtered = self._filter_pauses(audio_path, pauses_raw, segments)

        # Основные метрики
        words_per_minute = self._calculate_wpm(words_total, speaking_time_sec)
        speaking_ratio = speaking_time_sec / duration_sec if duration_sec > 0 else 0.0

        # Статистика
        filler_total, filler_detail = self._count_fillers(full_text)
        pauses_stats = self._summarize_pauses(pauses_filtered)
        phrase_stats = self._build_phrase_stats(segments, pauses_filtered)
>>>>>>> feature/gigachat-integration
        advice = self._generate_advice(
            words_per_minute, filler_total, words_total, pauses_stats, phrase_stats)

<<<<<<< HEAD
        # Дополнительные метрики с таймингами
        speech_windows = self._calculate_speech_windows(transcript)
        speaking_activity = self._build_speaking_activity(transcript)
        phrase_boundaries = self._extract_phrase_boundaries(
            transcript.segments)

        # Конвертация сегментов для JSON
        segments_dict = self._convert_segments_to_dict(transcript.segments)
=======
        # Статистика слов-паразитов
        filler_stats = FillerWordsStats(
            total=filler_total,
            per_100_words=round(
                (filler_total / words_total * 100), 1
            ) if words_total else 0.0,
            items=[
                {"word": name, "count": filler_detail.get(name, 0)}
                for name, _ in COMPILED_FILLERS
                if filler_detail.get(name, 0) > 0
            ],
        )
>>>>>>> feature/gigachat-integration

        return TimedAnalysisResult(
            duration_sec=round(duration_sec, 2),
            speaking_time_sec=round(speaking_time_sec, 2),
            speaking_ratio=round(speaking_ratio, 3),
            words_total=words_total,
            words_per_minute=round(words_per_minute, 1),

            filler_words=filler_stats,
            pauses=pauses_stats,
            phrases=phrase_stats,

            advice=advice,
<<<<<<< HEAD

            filler_words_detailed=filler_detailed,
            pauses_detailed=pauses_detailed,
            speech_rate_windows=speech_windows,
            emotional_peaks=[],  # TODO: добавить позже
            phrase_boundaries=phrase_boundaries,

            speaking_activity=speaking_activity,
            transcript_segments=segments_dict,
            transcript_text=transcript.text
        )

    # -------------------------------------------------
    # Основные методы расчета
    # -------------------------------------------------

    def _calculate_basic_metrics(
        self,
        transcript: Transcript
    ) -> Tuple[float, float, int]:
        """Рассчитывает базовые метрики."""
        segments = transcript.segments
        if not segments:
            return 0.0, 0.0, 0

        duration_sec = float(segments[-1].end)

        speaking_time_sec = 0.0
        for seg in segments:
            speaking_time_sec += max(0.0, seg.end - seg.start)

        words_total = len(self._split_words(transcript.text))

        return duration_sec, speaking_time_sec, words_total

    @staticmethod
    def _calculate_wpm(words_total: int, speaking_time_sec: float) -> float:
        """Рассчитывает слова в минуту."""
        if speaking_time_sec > 0 and words_total > 0:
            return words_total / (speaking_time_sec / 60.0)
        return 0.0

    # -------------------------------------------------
    # Слова-паразиты
    # -------------------------------------------------
=======
            transcript=full_text,
            gigachat_analysis=None,
        )

    def _empty_analysis_result(self, text: str) -> AnalysisResult:
        """Возвращает результат для пустого транскрипта"""
        return AnalysisResult(
            duration_sec=0.0,
            speaking_time_sec=0.0,
            speaking_ratio=0.0,
            words_total=0,
            words_per_minute=0.0,
            filler_words=FillerWordsStats(
                total=0,
                per_100_words=0.0,
                items=[]
            ),
            pauses=PausesStats(
                count=0,
                avg_sec=0.0,
                max_sec=0.0,
                long_pauses=[]
            ),
            phrases=PhraseStats(
                count=0,
                avg_words=0.0,
                avg_duration_sec=0.0,
                min_words=0,
                max_words=0,
                min_duration_sec=0.0,
                max_duration_sec=0.0,
                length_classification="insufficient_data",
                rhythm_variation="insufficient_data"
            ),
            advice=[
                AdviceItem(
                    category="speech_rate",
                    severity="warning",
                    title="Недостаточно данных",
                    observation="В аудио не обнаружено речи или транскрипт пуст",
                    recommendation="Попробуйте записать более четкую речь или увеличьте громкость"
                )
            ],
            transcript=text,
            gigachat_analysis=None,
        )

    @staticmethod
    def _calculate_speaking_stats(segments: List[TranscriptSegment]) -> Tuple[float, List[Dict[str, float]]]:
        """Рассчитывает время говорения и паузы"""
        speaking_time_sec = 0.0
        pauses_raw = []
        last_end = None

        for seg in segments:
            start = float(seg.start)
            end = float(seg.end)
            seg_duration = max(0.0, end - start)
            speaking_time_sec += seg_duration

            if last_end is not None:
                gap = start - last_end
                if gap >= MIN_PAUSE_GAP_SEC:
                    pauses_raw.append({
                        "start": last_end,
                        "end": start,
                        "duration": gap
                    })
            last_end = end

        return speaking_time_sec, pauses_raw

    def _filter_pauses(
        self,
        audio_path: Path | None,
        pauses: List[Dict[str, float]],
        segments: List[TranscriptSegment]
    ) -> List[Dict[str, float]]:
        """Фильтрует паузы по аудио"""
        if audio_path is None or not pauses:
            return pauses

        try:
            return self._filter_noisy_pauses(audio_path, pauses, segments)
        except Exception as e:
            # При ошибке чтения аудио возвращаем сырые паузы
            return pauses

    @staticmethod
    def _calculate_wpm(words_total: int, speaking_time_sec: float) -> float:
        """Рассчитывает слова в минуту"""
        if speaking_time_sec <= 0 or words_total <= 0:
            return 0.0
        return words_total / (speaking_time_sec / 60.0)
>>>>>>> feature/gigachat-integration

    @staticmethod
    def _split_words(text: str) -> List[str]:
        """Разделяет текст на слова"""
        if not text:
            return []
        return re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", text.lower())

    @staticmethod
    def _count_fillers(text: str) -> Tuple[int, Dict[str, int]]:
<<<<<<< HEAD
        """Подсчитывает слова-паразиты в тексте."""
=======
        """Считает слова-паразиты"""
        if not text:
            return 0, {}

>>>>>>> feature/gigachat-integration
        counts: Dict[str, int] = {}
        total = 0

        for name, pattern in COMPILED_FILLERS:
            matches = pattern.findall(text)
            if matches:
                count = len(matches)
                counts[name] = count
                total += count

        return total, counts

    def _build_filler_stats(
        self,
        total: int,
        detail: Dict[str, int],
        words_total: int
    ) -> FillerWordsStats:
        """Формирует статистику по словам-паразитам."""
        items = [
            {"word": name, "count": detail.get(name, 0)}
            for name, _ in COMPILED_FILLERS
        ]

        per_100_words = round(
            (total / words_total * 100), 1
        ) if words_total else 0.0

        return FillerWordsStats(
            total=total,
            per_100_words=per_100_words,
            items=items
        )

    def _find_fillers_with_timings(self, transcript: Transcript) -> List[TimedFillerWord]:
        """Находит слова-паразиты с точными таймингами."""
        results = []

        if not transcript.word_timings:
            return results

        for word_timing in transcript.word_timings:
            word_text = word_timing.word.lower().strip(",.!?;:()\"'")

            for filler_name, pattern in COMPILED_FILLERS:
                if pattern.match(word_text):
                    context = self._get_word_context(transcript, word_timing)
                    segment_start, segment_end = self._find_word_segment(
                        transcript, word_timing)

                    results.append(TimedFillerWord(
                        word=filler_name,
                        timestamp=round(word_timing.start, 2),
                        text_context=context,
                        segment_start=round(segment_start, 2),
                        segment_end=round(segment_end, 2)
                    ))
                    break

        return results

    # -------------------------------------------------
    # Паузы
    # -------------------------------------------------

    def _extract_raw_pauses(
        self,
        segments: List[TranscriptSegment]
    ) -> List[Dict[str, float]]:
        """Извлекает паузы между сегментами."""
        pauses = []
        for i in range(len(segments) - 1):
            gap = segments[i + 1].start - segments[i].end
            if gap >= MIN_PAUSE_GAP_SEC:
                pauses.append({
                    "start": segments[i].end,
                    "end": segments[i + 1].start,
                    "duration": gap
                })
        return pauses

    @staticmethod
    def _filter_noisy_pauses(
        audio_path: Path,
        pauses: List[Dict[str, float]],
        segments: List[TranscriptSegment],
    ) -> List[Dict[str, float]]:
<<<<<<< HEAD
        """Фильтрует шумные паузы по уровню громкости."""
        try:
            with wave.open(str(audio_path), "rb") as wf:
                n_channels, sampwidth, framerate, n_frames = wf.getparams()[:4]
=======
        """Фильтрует шумные паузы"""
        try:
            with wave.open(str(audio_path), "rb") as wf:
                n_channels, sampwidth, framerate, n_frames, *_ = wf.getparams()
>>>>>>> feature/gigachat-integration

                if n_channels != 1 or sampwidth != 2:
                    return pauses

                frames = wf.readframes(n_frames)
        except Exception:
            return pauses

        num_samples = len(frames) // 2
        if num_samples == 0:
            return pauses

        samples = struct.unpack("<{}h".format(num_samples), frames)

        def segment_rms(start_idx: int, end_idx: int) -> float:
            """Вычисляет RMS для сегмента"""
            count = end_idx - start_idx
            if count <= 0:
                return 0.0
<<<<<<< HEAD
=======

>>>>>>> feature/gigachat-integration
            sum_sq = sum(samples[i] * samples[i]
                         for i in range(start_idx, end_idx))
            return math.sqrt(sum_sq / count)

<<<<<<< HEAD
        # Громкость речи
        speech_rms_values: List[float] = []
        for seg in segments:
            start_idx = max(0, int(seg.start * framerate))
            end_idx = min(num_samples, int(seg.end * framerate))
=======
        # RMS речевых сегментов
        speech_rms_values = []
        for seg in segments:
            start_s = float(seg.start)
            end_s = float(seg.end)
            start_idx = max(0, int(start_s * framerate))
            end_idx = min(num_samples, int(end_s * framerate))

>>>>>>> feature/gigachat-integration
            if end_idx - start_idx < int(0.2 * framerate):
                continue

            r = segment_rms(start_idx, end_idx)
            if r > 0:
                speech_rms_values.append(r)

        if not speech_rms_values:
            return pauses

        # Медиана громкости речи
        speech_rms_values.sort()
        median_speech_rms = speech_rms_values[len(speech_rms_values) // 2]
        silence_threshold = median_speech_rms * SILENCE_FACTOR

        # Фильтрация пауз
        filtered = []
<<<<<<< HEAD
        for p in pauses:
            start_idx = max(0, int(p["start"] * framerate))
            end_idx = min(num_samples, int(p["end"] * framerate))
=======
        for pause in pauses:
            start_s = pause["start"]
            end_s = pause["end"]

            if end_s <= start_s:
                continue

            start_idx = max(0, int(start_s * framerate))
            end_idx = min(num_samples, int(end_s * framerate))

>>>>>>> feature/gigachat-integration
            if end_idx <= start_idx:
                continue

            r = segment_rms(start_idx, end_idx)
<<<<<<< HEAD
            if r < silence_threshold:
                filtered.append(p)
=======

            if r < silence_threshold:
                filtered.append(pause)
>>>>>>> feature/gigachat-integration

        return filtered

    @staticmethod
    def _summarize_pauses(pauses: List[Dict[str, float]]) -> PausesStats:
<<<<<<< HEAD
        """Суммирует статистику по паузам."""
=======
        """Суммирует статистику пауз"""
>>>>>>> feature/gigachat-integration
        if not pauses:
            return PausesStats(
                count=0,
                avg_sec=0.0,
                max_sec=0.0,
                long_pauses=[],
            )

        durations = [p["duration"] for p in pauses]
        avg_sec = sum(durations) / len(pauses)
        max_sec = max(durations)

        # Длинные паузы
        long_pauses = [
            {
                "start": round(p["start"], 2),
                "end": round(p["end"], 2),
                "duration": round(p["duration"], 2),
            }
            for p in pauses
            if p["duration"] >= LONG_PAUSE_SEC
        ]
<<<<<<< HEAD
=======

        # Топ-3 самых длинных
>>>>>>> feature/gigachat-integration
        long_pauses = sorted(
            long_pauses, key=lambda p: p["duration"], reverse=True)[:3]

        return PausesStats(
            count=len(pauses),
            avg_sec=round(avg_sec, 2),
            max_sec=round(max_sec, 2),
            long_pauses=long_pauses,
        )

    def _analyze_pauses_with_context(
        self,
        pauses: List[Dict[str, float]],
        segments: List[TranscriptSegment]
    ) -> List[TimedPause]:
        """Анализирует паузы с контекстом."""
        pauses_detailed = []

        for pause in pauses:
            segment_before = None
            segment_after = None

            for seg in segments:
                if abs(seg.end - pause["start"]) < 0.1:
                    segment_before = seg
                if abs(seg.start - pause["end"]) < 0.1:
                    segment_after = seg

            pause_type = self._classify_pause(pause["duration"])
            context_before = segment_before.text[-50:] if segment_before else ""
            context_after = segment_after.text[:50] if segment_after else ""

            pauses_detailed.append(TimedPause(
                start=round(pause["start"], 2),
                end=round(pause["end"], 2),
                duration=round(pause["duration"], 2),
                type=pause_type,
                context_before=context_before,
                context_after=context_after,
                speech_before_end=round(pause["start"], 2),
                speech_after_start=round(pause["end"], 2)
            ))

        return pauses_detailed

    @staticmethod
    def _classify_pause(duration: float) -> str:
        """Классифицирует паузу по длительности."""
        if duration < 1.0:
            return "natural"
        elif duration < 2.5:
            return "dramatic"
        elif duration < 5.0:
            return "long"
        else:
            return "awkward"

    # -------------------------------------------------
    # Фразы
    # -------------------------------------------------

    def _build_phrase_stats(
        self,
        segments: List[TranscriptSegment],
        pauses: List[Dict[str, float]],
    ) -> PhraseStats:
<<<<<<< HEAD
        """Анализирует структуру фраз."""
=======
        """Строит статистику фраз"""
>>>>>>> feature/gigachat-integration
        if not segments:
            return self._empty_phrase_stats()

        # Определяем границы фраз
<<<<<<< HEAD
        boundary_after_idx = set()
        for p in pauses:
            pause_start = p["start"]
=======
        boundary_after_idx = self._find_phrase_boundaries(segments, pauses)

        # Собираем фразы
        phrases_durations, phrases_words = self._collect_phrases(
            segments, boundary_after_idx)

        if not phrases_words:
            return self._empty_phrase_stats()

        return self._calculate_phrase_stats(phrases_durations, phrases_words)

    def _find_phrase_boundaries(
        self,
        segments: List[TranscriptSegment],
        pauses: List[Dict[str, float]]
    ) -> set:
        """Находит индексы сегментов, после которых есть паузы"""
        boundary_after_idx = set()

        for pause in pauses:
            pause_start = pause["start"]
>>>>>>> feature/gigachat-integration
            best_idx = None
            best_diff = float("inf")

            for i, seg in enumerate(segments):
                diff = abs(seg.end - pause_start)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i

            if best_idx is not None and best_diff <= PAUSE_SEGMENT_TIME_TOLERANCE:
                boundary_after_idx.add(best_idx)

<<<<<<< HEAD
        # Разделяем на фразы
=======
        return boundary_after_idx

    def _collect_phrases(
        self,
        segments: List[TranscriptSegment],
        boundary_after_idx: set
    ) -> Tuple[List[float], List[int]]:
        """Собирает фразы из сегментов"""
>>>>>>> feature/gigachat-integration
        phrases_durations: List[float] = []
        phrases_words: List[int] = []
        phrase_start_idx = 0

        for idx in range(len(segments)):
            is_boundary = idx in boundary_after_idx
            is_last_seg = idx == len(segments) - 1

            if is_boundary or is_last_seg:
                phrase_end_idx = idx
                segs = segments[phrase_start_idx: phrase_end_idx + 1]
<<<<<<< HEAD
=======

>>>>>>> feature/gigachat-integration
                if segs:
                    dur, wcount = self._phrase_metrics(segs)
                    if wcount > 0 and dur > 0:
                        phrases_durations.append(dur)
                        phrases_words.append(wcount)

                phrase_start_idx = idx + 1

<<<<<<< HEAD
            if is_last_seg and phrase_start_idx <= idx:
                # закрываем последнюю фразу (если не закрыли выше)
                segs = segments[phrase_start_idx: idx + 1]
                if segs:
                    dur, wcount = self._phrase_metrics(segs)
                    if wcount > 0 and dur > 0:
                        phrases_durations.append(dur)
                        phrases_words.append(wcount)

        if not phrases_words:
            return PhraseStats(
                count=0,
                avg_words=0.0,
                avg_duration_sec=0.0,
                min_words=0,
                max_words=0,
                min_duration_sec=0.0,
                max_duration_sec=0.0,
                length_classification="insufficient_data",
                rhythm_variation="insufficient_data",
            )

        # Статистика по фразам
=======
        return phrases_durations, phrases_words

    def _calculate_phrase_stats(
        self,
        phrases_durations: List[float],
        phrases_words: List[int]
    ) -> PhraseStats:
        """Рассчитывает статистику фраз"""
>>>>>>> feature/gigachat-integration
        count = len(phrases_words)
        avg_words = sum(phrases_words) / count
        avg_dur = sum(phrases_durations) / count

<<<<<<< HEAD
        # Классификация
=======
        # Классификация по длине фраз
>>>>>>> feature/gigachat-integration
        if avg_words < 8:
            length_class = "short_phrases"
        elif avg_words <= 25:
            length_class = "balanced"
        else:
            length_class = "long_phrases"

        # Вариативность ритма
<<<<<<< HEAD
        if count < 2 or avg_dur <= 0:
            rhythm_var = "insufficient_data"
        else:
            mean_dur = avg_dur
            var = sum((d - mean_dur) ** 2 for d in phrases_durations) / count
            std = math.sqrt(var)
            cv = std / mean_dur
            if cv < 0.25:
                rhythm_var = "uniform"
            elif cv < 0.6:
                rhythm_var = "moderately_variable"
            else:
                rhythm_var = "highly_variable"
=======
        rhythm_var = self._calculate_rhythm_variation(
            phrases_durations, avg_dur, count)
>>>>>>> feature/gigachat-integration

        return PhraseStats(
            count=count,
            avg_words=round(avg_words, 1),
            avg_duration_sec=round(avg_dur, 2),
            min_words=min(phrases_words),
            max_words=max(phrases_words),
            min_duration_sec=round(min(phrases_durations), 2),
            max_duration_sec=round(max(phrases_durations), 2),
            length_classification=length_class,
            rhythm_variation=rhythm_var,
        )

<<<<<<< HEAD
    def _process_phrase(
        self,
        segments: List[TranscriptSegment],
        start_idx: int,
        end_idx: int,
        durations: List[float],
        word_counts: List[int],
    ) -> None:
        """Обрабатывает одну фразу и добавляет ее метрики."""
        segs = segments[start_idx:end_idx + 1]
        if segs:
            dur, wcount = self._phrase_metrics(segs)
            if wcount > 0 and dur > 0:
                durations.append(dur)
                word_counts.append(wcount)
=======
    @staticmethod
    def _calculate_rhythm_variation(
        durations: List[float],
        mean_dur: float,
        count: int
    ) -> str:
        """Рассчитывает вариативность ритма"""
        if count < 2 or mean_dur <= 0:
            return "insufficient_data"

        variance = sum((d - mean_dur) ** 2 for d in durations) / count
        std = math.sqrt(variance)
        cv = std / mean_dur

        if cv < 0.25:
            return "uniform"
        elif cv < 0.6:
            return "moderately_variable"
        else:
            return "highly_variable"

    @staticmethod
    def _empty_phrase_stats() -> PhraseStats:
        """Возвращает пустую статистику фраз"""
        return PhraseStats(
            count=0,
            avg_words=0.0,
            avg_duration_sec=0.0,
            min_words=0,
            max_words=0,
            min_duration_sec=0.0,
            max_duration_sec=0.0,
            length_classification="insufficient_data",
            rhythm_variation="insufficient_data",
        )
>>>>>>> feature/gigachat-integration

    def _phrase_metrics(
        self,
        segs: List[TranscriptSegment],
    ) -> Tuple[float, int]:
<<<<<<< HEAD
        """Возвращает длительность и количество слов во фразе."""
        if not segs:
            return 0.0, 0
        duration = segs[-1].end - segs[0].start
=======
        """Возвращает длительность и количество слов во фразе"""
        if not segs:
            return 0.0, 0

        start = float(segs[0].start)
        end = float(segs[-1].end)
        duration = max(0.0, end - start)

>>>>>>> feature/gigachat-integration
        text = " ".join(s.text for s in segs)
        words = self._split_words(text)

        return duration, len(words)

    # -------------------------------------------------
    # Советы
    # -------------------------------------------------

    @staticmethod
    def _generate_advice(
        words_per_minute: float,
        filler_total: int,
        words_total: int,
        pauses_stats: PausesStats,
        phrase_stats: PhraseStats,
    ) -> List[AdviceItem]:
<<<<<<< HEAD
        """Генерирует советы по улучшению речи."""
        advice = []

        # 1. Темп речи
        if words_total == 0 or words_per_minute == 0:
            speech_obs = "Автоматический анализ темпа речи затруднён."
            speech_rec = "Запишите более продолжительный фрагмент с отчётливой речью."
            speech_sev = "info"
        elif words_per_minute < MIN_COMFORT_WPM:
            speech_obs = f"Темп речи примерно {words_per_minute:.1f} слов в минуту, ниже типичного диапазона ({
                MIN_COMFORT_WPM:.0f}–{MAX_COMFORT_WPM:.0f})."
            speech_rec = "Ускорьте речь, сократив избыточные паузы."
            speech_sev = "suggestion"
        elif words_per_minute > MAX_COMFORT_WPM:
            speech_obs = f"Темп речи примерно {words_per_minute:.1f} слов в минуту, выше типичного диапазона ({
                MIN_COMFORT_WPM:.0f}–{MAX_COMFORT_WPM:.0f})."
            speech_rec = "Замедлите подачу, делая более заметные паузы."
            speech_sev = "suggestion"
        else:
            speech_observation = (
                "Оценённый темп речи составляет примерно {wpm:.1f} слов в минуту, "
                "что находится в пределах типичного диапазона публичных выступлений."
            ).format(wpm=words_per_minute)
            speech_recommendation = (
                "Сохраняйте выбранный темп и при необходимости варьируйте его для "
                "подчёркивания ключевых смысловых блоков."
            )
            speech_severity = "info"
=======
        """Генерирует рекомендации"""
        advice = []

        # 1. Темп речи
        advice.append(SpeechAnalyzer._generate_speech_rate_advice(
            words_per_minute, words_total))
>>>>>>> feature/gigachat-integration

        # 2. Слова-паразиты
        advice.append(SpeechAnalyzer._generate_filler_advice(
            filler_total, words_total))

<<<<<<< HEAD
        # --- 2. Слова-паразиты ---
        fillers_per_100 = (filler_total / words_total *
                           100) if words_total else 0.0

        if filler_total == 0:
            filler_obs = "Не обнаружено слов-паразитов."
            filler_rec = "Отличный контроль речи!"
            filler_sev = "info"
        elif fillers_per_100 <= 3:
            filler_obs = f"Слова-паразиты присутствуют, но их доля невелика ({
                fillers_per_100:.1f} на 100 слов)."
            filler_rec = "Можно дополнительно снизить их количество."
            filler_sev = "info"
        elif fillers_per_100 <= 8:
            filler_obs = f"Доля слов-паразитов {
                fillers_per_100:.1f} на 100 слов."
            filler_rec = "Обратите внимание на часто повторяющиеся конструкции."
            filler_sev = "suggestion"
        else:
            filler_obs = f"Высокая доля слов-паразитов ({
                fillers_per_100:.1f} на 100 слов)."
            filler_rec = "Тренируйтесь делать сознательные паузы вместо слов-паразитов."
            filler_sev = "warning"

        advice.append(AdviceItem(
            category="filler_words",
            severity=filler_sev,
            title="Слова-паразиты",
            observation=filler_obs,
            recommendation=filler_rec,
        ))

        # 3. Паузы
        if pauses_stats.count == 0:
            pauses_obs = "Практически отсутствуют выделенные паузы."
            pauses_rec = "Используйте короткие паузы для выделения ключевых мыслей."
            pauses_sev = "info"
        else:
            long_count = len(pauses_stats.long_pauses)
            long_fraction = long_count / pauses_stats.count if pauses_stats.count > 0 else 0.0

            if long_count > 0 and long_fraction > 0.3:
                pauses_obs = f"Обнаружены длинные паузы (до {
                    pauses_stats.max_sec:.1f} секунд)."
                pauses_rec = "Заполняйте длинные паузы чёткими вводными фразами."
                pauses_sev = "suggestion"
            else:
                pauses_obs = f"Паузы присутствуют (средняя длительность {
                    pauses_stats.avg_sec:.1f} секунд)."
                pauses_rec = "Баланс пауз выглядит естественным."
                pauses_sev = "info"

        advice.append(AdviceItem(
            category="pauses",
            severity=pauses_sev,
            title="Паузы в речи",
            observation=pauses_obs,
            recommendation=pauses_rec,
        ))

        # 4. Структура фраз
        if phrase_stats.count <= 1:
            phr_obs = "Анализ структуры фраз затруднён."
            phr_rec = "Используйте паузы между смысловыми блоками."
            phr_sev = "info"
        else:
            phr_obs = f"Средняя длина фразы: {
                phrase_stats.avg_words:.1f} слов."

            if phrase_stats.length_classification == "short_phrases":
                phr_obs += " Фразы в основном короткие."
                phr_rec = "Объединяйте близкие по смыслу предложения."
                phr_sev = "suggestion"
            elif phrase_stats.length_classification == "long_phrases":
                phr_obs += " Фразы достаточно длинные."
                phr_rec = "Разбивайте длинные фразы на смысловые единицы."
                phr_sev = "suggestion"
            else:
                phr_obs += " Длина фраз сбалансирована."
                phr_rec = "Сохраняйте текущую структуру."
                phr_sev = "info"

        advice.append(AdviceItem(
            category="phrasing",
            severity=phr_sev,
            title="Структура фраз",
            observation=phr_obs,
            recommendation=phr_rec,
        ))

        return advice

    # -------------------------------------------------
    # Дополнительные метрики с таймингами
    # -------------------------------------------------

    def _calculate_speech_windows(
        self,
        transcript: Transcript,
        window_size: float = 30.0,
        step: float = 15.0
    ) -> List[SpeechRateWindow]:
        """Рассчитывает темп речи в скользящих окнах."""
        windows = []

        if not transcript.segments:
            return windows

        total_duration = transcript.segments[-1].end
        current_start = 0.0

        while current_start < total_duration:
            window_end = min(current_start + window_size, total_duration)

            words_in_window = 0
            speaking_time = 0.0

            for segment in transcript.segments:
                seg_start = max(segment.start, current_start)
                seg_end = min(segment.end, window_end)

                if seg_end > seg_start:
                    overlap = seg_end - seg_start
                    speaking_time += overlap

                    total_seg = segment.end - segment.start
                    if total_seg > 0:
                        ratio = overlap / total_seg
                        seg_words = len(self._split_words(segment.text))
                        words_in_window += int(seg_words * ratio)

            wpm = words_in_window / \
                (speaking_time / 60.0) if speaking_time > 0 else 0.0

            windows.append(SpeechRateWindow(
                window_start=round(current_start, 2),
                window_end=round(window_end, 2),
                word_count=words_in_window,
                words_per_minute=round(wpm, 1),
                speaking_time=round(speaking_time, 2)
            ))

            current_start += step

        return windows

    def _build_speaking_activity(
        self,
        transcript: Transcript,
        resolution: float = 1.0  # Оптимизировано: точка каждую секунду
    ) -> List[Dict[str, float]]:
        """Создает массив активности говорения."""
        activity = []

        if not transcript.segments:
            return activity

        total_duration = transcript.segments[-1].end
        current_time = 0.0

        while current_time <= total_duration:
            is_speaking = 0.0

            for segment in transcript.segments:
                if segment.start <= current_time <= segment.end:
                    is_speaking = 1.0
                    break

            activity.append({
                "time": round(current_time, 2),
                "is_speaking": is_speaking
            })

            current_time += resolution

        return activity

    @staticmethod
    def _extract_phrase_boundaries(segments: List[TranscriptSegment]) -> List[float]:
        """Извлекает границы фраз."""
        return [round(seg.start, 2) for seg in segments]

    @staticmethod
    def _convert_segments_to_dict(segments: List[TranscriptSegment]) -> List[Dict[str, Any]]:
        """Конвертирует сегменты в словари для JSON."""
        return [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "words": [{"word": w.word, "start": w.start, "end": w.end} for w in s.words]
            }
            for s in segments
        ]

    # -------------------------------------------------
    # Вспомогательные методы
    # -------------------------------------------------

    def _get_word_context(
        self,
        transcript: Transcript,
        word_timing: WordTiming,
        words_before: int = 2,
        words_after: int = 2
    ) -> str:
        """Возвращает контекст слова."""
        for segment in transcript.segments:
            if segment.start <= word_timing.start <= segment.end:
                all_words = segment.text.split()

                word_index = -1
                for i, word in enumerate(all_words):
                    if word_timing.word.lower() in word.lower():
                        word_index = i
                        break

                if word_index >= 0:
                    start_idx = max(0, word_index - words_before)
                    end_idx = min(len(all_words), word_index + words_after + 1)
                    context_words = all_words[start_idx:end_idx]

                    marked_words = []
                    for i, w in enumerate(context_words):
                        if start_idx + i == word_index:
                            marked_words.append(f"[{w}]")
                        else:
                            marked_words.append(w)

                    return " ".join(marked_words)

        return f"[{word_timing.word}]"

    def _find_word_segment(
        self,
        transcript: Transcript,
        word_timing: WordTiming
    ) -> tuple[float, float]:
        """Находит сегмент, содержащий слово."""
        for segment in transcript.segments:
            if segment.start <= word_timing.start <= segment.end:
                return segment.start, segment.end

        return word_timing.start, word_timing.end
=======
        # 3. Паузы
        advice.append(SpeechAnalyzer._generate_pauses_advice(pauses_stats))

        # 4. Структура фраз
        advice.append(SpeechAnalyzer._generate_phrasing_advice(phrase_stats))

        return advice

    @staticmethod
    def _generate_speech_rate_advice(wpm: float, words_total: int) -> AdviceItem:
        """Генерирует рекомендации по темпу речи"""
        if words_total == 0 or wpm == 0:
            return AdviceItem(
                category="speech_rate",
                severity="info",
                title="Темп речи",
                observation="Не удалось определить темп речи",
                recommendation="Запишите более продолжительный фрагмент с отчётливой речью"
            )

        if wpm < MIN_COMFORT_WPM:
            return AdviceItem(
                category="speech_rate",
                severity="suggestion",
                title="Темп речи",
                observation=f"Темп речи составляет примерно {wpm:.1f} слов в минуту, что ниже типичного диапазона ({
                    MIN_COMFORT_WPM:.0f}–{MAX_COMFORT_WPM:.0f} слов в минуту)",
                recommendation="Попробуйте немного ускорить речь, сокращая избыточные паузы"
            )
        elif wpm > MAX_COMFORT_WPM:
            return AdviceItem(
                category="speech_rate",
                severity="suggestion",
                title="Темп речи",
                observation=f"Темп речи составляет примерно {wpm:.1f} слов в минуту, что выше типичного диапазона ({
                    MIN_COMFORT_WPM:.0f}–{MAX_COMFORT_WPM:.0f} слов в минуту)",
                recommendation="Попробуйте замедлить речь, делая более заметные паузы между фразами"
            )
        else:
            return AdviceItem(
                category="speech_rate",
                severity="info",
                title="Темп речи",
                observation=f"Темп речи составляет примерно {
                    wpm:.1f} слов в минуту, что находится в пределах нормы",
                recommendation="Сохраняйте выбранный темп и варьируйте его для выделения ключевых мыслей"
            )

    @staticmethod
    def _generate_filler_advice(filler_total: int, words_total: int) -> AdviceItem:
        """Генерирует рекомендации по словам-паразитам"""
        fillers_per_100 = (filler_total / words_total *
                           100) if words_total else 0.0

        if filler_total == 0:
            return AdviceItem(
                category="filler_words",
                severity="info",
                title="Слова-паразиты",
                observation="Слов-паразитов не обнаружено",
                recommendation="Продолжайте контролировать свою речь"
            )
        elif fillers_per_100 <= 3:
            return AdviceItem(
                category="filler_words",
                severity="info",
                title="Слова-паразиты",
                observation=f"Слова-паразиты присутствуют в небольшом количестве ({
                    fillers_per_100:.1f} на 100 слов)",
                recommendation="Старайтесь заменять слова-паразиты короткими паузами"
            )
        elif fillers_per_100 <= 8:
            return AdviceItem(
                category="filler_words",
                severity="suggestion",
                title="Слова-паразиты",
                observation=f"Заметное количество слов-паразитов ({
                    fillers_per_100:.1f} на 100 слов)",
                recommendation="Обратите внимание на наиболее частые слова-паразиты и сознательно избегайте их"
            )
        else:
            return AdviceItem(
                category="filler_words",
                severity="warning",
                title="Слова-паразиты",
                observation=f"Высокий уровень слов-паразитов ({
                    fillers_per_100:.1f} на 100 слов)",
                recommendation="Потренируйтесь говорить, сознательно избегая слов-паразитов, используйте паузы вместо них"
            )

    @staticmethod
    def _generate_pauses_advice(pauses_stats: PausesStats) -> AdviceItem:
        """Генерирует рекомендации по паузам"""
        if pauses_stats.count == 0:
            return AdviceItem(
                category="pauses",
                severity="info",
                title="Паузы в речи",
                observation="В речи практически отсутствуют паузы",
                recommendation="Используйте короткие паузы для структурирования речи"
            )

        long_count = len(pauses_stats.long_pauses)
        if long_count > 0 and long_count / pauses_stats.count > 0.3:
            return AdviceItem(
                category="pauses",
                severity="suggestion",
                title="Паузы в речи",
                observation=f"Обнаружены длинные паузы (до {
                    pauses_stats.max_sec:.1f} секунд)",
                recommendation="Сократите слишком длинные паузы или заполните их связующими фразами"
            )
        else:
            return AdviceItem(
                category="pauses",
                severity="info",
                title="Паузы в речи",
                observation=f"Паузы используются естественно (средняя длина {
                    pauses_stats.avg_sec:.1f} секунд)",
                recommendation="Продолжайте использовать паузы для лучшего восприятия речи"
            )

    @staticmethod
    def _generate_phrasing_advice(phrase_stats: PhraseStats) -> AdviceItem:
        """Генерирует рекомендации по структуре фраз"""
        if phrase_stats.count <= 1:
            return AdviceItem(
                category="phrasing",
                severity="info",
                title="Структура фраз",
                observation="Недостаточно данных для анализа структуры фраз",
                recommendation="Используйте паузы для разделения смысловых блоков"
            )

        observation = f"Средняя длина фразы: {
            phrase_stats.avg_words:.1f} слов ({phrase_stats.avg_duration_sec:.1f} секунд)"
        recommendation = ""
        severity = "info"

        if phrase_stats.length_classification == "short_phrases":
            observation += ". Фразы довольно короткие"
            recommendation = "Объединяйте связанные мысли в более длинные фразы"
            severity = "suggestion"
        elif phrase_stats.length_classification == "long_phrases":
            observation += ". Фразы довольно длинные"
            recommendation = "Разбивайте сложные мысли на более короткие фразы"
            severity = "suggestion"
        else:
            observation += ". Длина фраз сбалансирована"
            recommendation = "Продолжайте использовать фразы разной длины для поддержания ритма"

        return AdviceItem(
            category="phrasing",
            severity=severity,
            title="Структура фраз",
            observation=observation,
            recommendation=recommendation
        )
>>>>>>> feature/gigachat-integration
