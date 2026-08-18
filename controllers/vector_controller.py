import logging

from fastapi import APIRouter, Depends, HTTPException

from auth import get_current_user
from services.pdf_service import AskPDFRequest, ask_pdf_endpoint
from utils.state import task_status

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/vector/history")
async def get_vector_history(request_user=Depends(get_current_user)):
    """Get history of vector indexes (Qdrant collections) for the current user."""
    logger.info(f"Vector index history endpoint called by user: {request_user.get('user_id')}")
    user_id = request_user.get("user_id")

    try:
        try:
            from services.qdrant_service import get_task_collections

            qdrant_collections = get_task_collections(user_id)
            if qdrant_collections:
                history = []
                for coll in qdrant_collections:
                    history.append({
                        "key": f"{user_id}/qdrant/{coll['task_id']}",
                        "task_id": coll["task_id"],
                        "created_at": "",
                        "size": coll.get("points_count", 0),
                    })
                return {"history": history}
        except Exception as qdrant_error:
            logger.error(f"Qdrant query failed: {str(qdrant_error)}")

        logger.error("Qdrant is required for vector storage. Please ensure Qdrant is running.")
        return {"history": []}
    except Exception as exc:
        logger.error(f"Error retrieving vector index history: {str(exc)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve vector index history: {str(exc)}") from exc


@router.post("/vector/query/{task_id}")
async def query_task_vector(
    task_id: str,
    question: str,
    request_user=Depends(get_current_user),
):
    """Query a specific task's vector embeddings (Qdrant)."""
    request_data = AskPDFRequest(question=question, task_ids=[task_id])

    try:
        result = await ask_pdf_endpoint(request_data, request_user, task_status={}, user_tasks={})
        return result
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Vector index not found for task_id: {task_id}. Error: {str(exc)}") from exc
