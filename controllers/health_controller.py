import logging

from fastapi import APIRouter

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/")
def health():
    logger.info("Health check endpoint called")
    return {"status": "ok", "message": "VectraMind API is running"}
