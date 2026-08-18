import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from auth import get_current_user
from services.kg_service import (
    get_knowledge_graph_history,
    get_node_explanation,
    get_user_knowledge_graph,
    get_user_task_ids,
)

router = APIRouter()
logger = logging.getLogger(__name__)


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
        return get_user_knowledge_graph(user_id, task_id=task_id, limit=limit)
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
