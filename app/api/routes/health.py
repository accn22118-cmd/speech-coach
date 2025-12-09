from fastapi import APIRouter, Depends
from app.core.config import settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Speech Coach API",
        "version": "1.0.0",
        "features": {
            "whisper": True,
            "gigachat": settings.gigachat_enabled
        }
    }
