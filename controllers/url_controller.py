import logging

from fastapi import APIRouter, BackgroundTasks, Depends, Request

from auth import get_current_user
from services.url_service import ask_endpoint, initialize_vector_index_endpoint
from utils.state import file_path, task_status

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/initialize_vector_index")
async def initialize_vector_index(
    background_tasks: BackgroundTasks,
    request: Request,
    request_user=Depends(get_current_user),
):
    logger.info("Initialize vector index endpoint called (Qdrant)")
    user_id = request_user.get("user_id", "default")
    return await initialize_vector_index_endpoint(background_tasks, request, task_status, file_path, user_id)


@router.post("/ask")
async def ask(request: Request):
    logger.info("Ask endpoint called")
    return await ask_endpoint(request, file_path)
