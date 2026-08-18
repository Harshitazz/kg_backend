import logging
from typing import Dict, List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile

from auth import get_current_user
from services.pdf_service import AskPDFRequest, ask_pdf_endpoint, upload_pdfs_endpoint
from utils.state import file_path, task_status, user_tasks

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/task_status/{task_id}")
def get_task_status(task_id: str):
    status = task_status.get(task_id, "Not found")
    logger.info(f"Task status check: {task_id} = {status}")
    return {"task_id": task_id, "status": status}


@router.post("/upload_pdfs")
@router.post("/upload_pdfs/")
async def upload_pdfs(
    background_tasks: BackgroundTasks,
    files: list[UploadFile],
    request_user=Depends(get_current_user),
):
    logger.info(f"Upload PDFs endpoint called by user: {request_user.get('user_id')}")
    logger.info(f"Received {len(files)} file(s)")
    return await upload_pdfs_endpoint(background_tasks, files, task_status, request_user, user_tasks)


@router.post("/ask_pdf")
async def ask_pdf(
    request: AskPDFRequest,
    request_user=Depends(get_current_user),
):
    logger.info(f"Ask PDF endpoint called by user: {request_user.get('user_id')}")
    logger.info(f"Question: {request.question}")

    user_id = request_user.get("user_id")
    user_task_ids = user_tasks.get(user_id, [])

    active_tasks = []
    for task_id in user_task_ids:
        status = task_status.get(task_id, "")
        if status in ["Pending", "Processing"]:
            active_tasks.append(task_id)

    if active_tasks:
        raise HTTPException(
            status_code=409,
            detail={
                "message": f"PDF processing is still in progress. Please wait for task(s) {', '.join(active_tasks[:3])} to complete before asking questions.",
                "task_id": active_tasks[0],
                "task_ids": active_tasks,
                "status_url": f"/task_status/{active_tasks[0]}",
            },
        )

    return await ask_pdf_endpoint(request, request_user, task_status, user_tasks)
