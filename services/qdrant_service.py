"""Compatibility wrapper around the Qdrant repository layer."""

import logging

from repositories.qdrant_repository import (
    create_vectorstore_for_task,
    get_collection_name,
    get_embedding_model,
    get_qdrant_client,
    get_task_collections,
    query_vectorstore_multi_task,
)

logger = logging.getLogger(__name__)

__all__ = [
    "get_qdrant_client",
    "get_embedding_model",
    "get_collection_name",
    "create_vectorstore_for_task",
    "query_vectorstore_multi_task",
    "get_task_collections",
    "delete_collection",
]


def delete_collection(user_id: str, task_id: str) -> bool:
    """Delete a Qdrant collection for a task."""
    try:
        client = get_qdrant_client()
        collection_name = get_collection_name(user_id, task_id)
        client.delete_collection(collection_name)
        logger.info(f"Deleted Qdrant collection: {collection_name}")
        return True
    except Exception as e:
        logger.error(f"Error deleting collection {collection_name}: {str(e)}")
        return False
