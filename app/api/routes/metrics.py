import logging
from fastapi import APIRouter, HTTPException
from pathlib import Path
import json
from datetime import datetime, timedelta

router = APIRouter(prefix="/api/v1", tags=["metrics"])
logger = logging.getLogger(__name__)


@router.get("/metrics/summary")
async def get_metrics_summary(hours: int = 24):
    """
    Возвращает сводку метрик за последние N часов.
    """
    try:
        metrics_file = Path("logs/metrics.jsonl")
        if not metrics_file.exists():
            return {
                "total_processed": 0,
                "success_rate": 0,
                "avg_processing_time": 0,
                "data": []
            }

        cutoff_time = datetime.now() - timedelta(hours=hours)

        metrics = []
        with open(metrics_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    metric = json.loads(line.strip())
                    metric_time = datetime.fromisoformat(metric["timestamp"])

                    if metric_time >= cutoff_time:
                        metrics.append(metric)
                except json.JSONDecodeError:
                    continue

        if not metrics:
            return {
                "total_processed": 0,
                "success_rate": 0,
                "avg_processing_time": 0,
                "data": []
            }

        # Рассчитываем статистику
        total = len(metrics)
        successful = sum(1 for m in metrics if m.get("success", False))
        avg_time = sum(m.get("processing_time_sec", 0)
                       for m in metrics) / total

        return {
            "total_processed": total,
            "success_rate": round(successful / total * 100, 1) if total > 0 else 0,
            "avg_processing_time": round(avg_time, 2),
            "avg_file_size_mb": round(sum(m.get("file_size_mb", 0) for m in metrics) / total, 2),
            "data": metrics[-50:]  # Последние 50 записей
        }

    except Exception as e:
        logger.error(f"Ошибка получения метрик: {e}")
        raise HTTPException(status_code=500, detail="Ошибка получения метрик")


@router.get("/metrics/system")
async def get_system_metrics():
    """
    Возвращает текущие системные метрики.
    """
    try:
        import psutil
        import platform

        return {
            "cpu": {
                "percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
                "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None
            },
            "memory": {
                "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                "percent": psutil.virtual_memory().percent
            },
            "disk": {
                "total_gb": round(psutil.disk_usage("/").total / (1024**3), 2),
                "used_gb": round(psutil.disk_usage("/").used / (1024**3), 2),
                "percent": psutil.disk_usage("/").percent
            },
            "system": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "uptime_seconds": int(psutil.boot_time())
            },
            "process": {
                "memory_mb": round(psutil.Process().memory_info().rss / (1024**2), 2),
                "cpu_percent": psutil.Process().cpu_percent(),
                "threads": psutil.Process().num_threads()
            }
        }

    except ImportError:
        raise HTTPException(
            status_code=500, detail="Требуется установка psutil")
    except Exception as e:
        logger.error(f"Ошибка получения системных метрик: {e}")
        raise HTTPException(
            status_code=500, detail="Ошибка получения системных метрик")
