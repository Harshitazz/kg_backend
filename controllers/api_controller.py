import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, UploadFile

from auth import get_current_user
from services.kg_service import (
    get_knowledge_graph_history,
    get_node_explanation,
    get_user_knowledge_graph,
    get_user_task_ids,
)
from services.pdf_service import AskPDFRequest, ask_pdf_endpoint, upload_pdfs_endpoint
from services.url_service import initialize_vector_index_endpoint, ask_endpoint
from utils.common import normalize_task_ids
from utils.state import file_path, task_status, user_tasks

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/")
def health():
    logger.info("Health check endpoint called")
    return {"status": "ok", "message": "VectraMind API is running"}


@router.post("/initialize_vector_index")
async def initialize_vector_index(
    background_tasks: BackgroundTasks,
    request: Request,
    request_user=Depends(get_current_user),
):
    logger.info("Initialize vector index endpoint called (Qdrant)")
    user_id = request_user.get("user_id", "default")
    return await initialize_vector_index_endpoint(background_tasks, request, task_status, file_path, user_id)


@router.get("/task_status/{task_id}")
def get_task_status(task_id: str):
    status = task_status.get(task_id, "Not found")
    logger.info(f"Task status check: {task_id} = {status}")
    return {"task_id": task_id, "status": status}


@router.post("/ask")
async def ask(request: Request):
    logger.info("Ask endpoint called")
    return await ask_endpoint(request, file_path)


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


@router.get("/knowledge_graph")
async def get_knowledge_graph(
    request_user=Depends(get_current_user),
    task_id: Optional[str] = None,
    limit: int = 50,
):
    """Get knowledge graph data for the current user, optionally filtered by task_id."""
    logger.info(f"Knowledge graph endpoint called by user: {request_user.get('user_id')}, task_id: {task_id}")
    user_id = request_user.get("user_id")

    try:
        graph_data = get_user_knowledge_graph(user_id, task_id=task_id, limit=limit)
        return graph_data
    except Exception as exc:
        logger.error(f"Error retrieving knowledge graph: {str(exc)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve knowledge graph: {str(exc)}") from exc


@router.get("/knowledge_graph/tasks")
async def get_knowledge_graph_tasks(request_user=Depends(get_current_user)):
    """Get list of task_ids that have knowledge graphs for the current user."""
    logger.info(f"Knowledge graph tasks endpoint called by user: {request_user.get('user_id')}")
    user_id = request_user.get("user_id")

    try:
        task_ids = get_user_task_ids(user_id)
        return {"task_ids": task_ids}
    except Exception as exc:
        logger.error(f"Error retrieving task IDs: {str(exc)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve task IDs: {str(exc)}") from exc


@router.get("/knowledge_graph/history")
async def get_knowledge_graph_history_endpoint(request_user=Depends(get_current_user)):
    """Get history of knowledge graphs for the current user."""
    logger.info(f"Knowledge graph history endpoint called by user: {request_user.get('user_id')}")
    user_id = request_user.get("user_id")

    try:
        history = get_knowledge_graph_history(user_id)
        return {"history": history}
    except Exception as exc:
        logger.error(f"Error retrieving KG history: {str(exc)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve KG history: {str(exc)}") from exc


@router.get("/knowledge_graph/node/explain")
async def explain_node(
    node_name: str,
    task_id: Optional[str] = None,
    request_user=Depends(get_current_user),
):
    """Get explanation for a specific node."""
    logger.info(f"Node explanation requested for: {node_name}, user: {request_user.get('user_id')}")
    user_id = request_user.get("user_id")

    try:
        explanation = get_node_explanation(node_name, user_id, task_id=task_id)
        return {"node_name": node_name, "explanation": explanation}
    except Exception as exc:
        logger.error(f"Error generating node explanation: {str(exc)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate explanation: {str(exc)}") from exc


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
    from services.pdf_service import AskPDFRequest, ask_pdf_endpoint

    user_id = request_user.get("user_id")
    request_data = AskPDFRequest(question=question, task_ids=[task_id])

    try:
        result = await ask_pdf_endpoint(request_data, request_user, task_status={}, user_tasks={})
        return result
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Vector index not found for task_id: {task_id}. Error: {str(exc)}") from exc

