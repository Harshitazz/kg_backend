"""
Qdrant Vector Store repository for persistent vector storage.
"""
import logging
import os
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    VectorParams,
)

logger = logging.getLogger(__name__)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

_qdrant_client = None
_embedding_model = None


def get_qdrant_client() -> QdrantClient:
    """Get or create Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        try:
            _qdrant_client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
            )
            _qdrant_client.get_collections()
            logger.info(f"Connected to Qdrant at {QDRANT_URL}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            raise
    return _qdrant_client


def get_embedding_model():
    """Get the shared embedding model."""
    global _embedding_model
    if _embedding_model is None:
        from services.pdf_service import get_embedding_model

        _embedding_model = get_embedding_model()
    return _embedding_model


def get_collection_name(user_id: str, task_id: str) -> str:
    """Generate collection name for user and task."""
    safe_user_id = user_id.replace("-", "_").replace("/", "_")
    safe_task_id = task_id.replace("-", "_").replace("/", "_")
    return f"{safe_user_id}_{safe_task_id}"


def create_vectorstore_for_task(
    documents: List[Document],
    user_id: str,
    task_id: str,
    source: str,
) -> QdrantVectorStore:
    """Create Qdrant vectorstore for a specific task."""
    client = get_qdrant_client()
    embeddings = get_embedding_model()
    collection_name = get_collection_name(user_id, task_id)

    for doc in documents:
        if not doc.metadata:
            doc.metadata = {}
        doc.metadata["user_id"] = user_id
        doc.metadata["task_id"] = task_id
        if "source" not in doc.metadata:
            doc.metadata["source"] = source

    try:
        test_embedding = embeddings.embed_query("test")
        embedding_dim = len(test_embedding)
        logger.debug(f"Detected embedding dimension: {embedding_dim}")
    except Exception as e:
        embedding_dim = 384
        logger.warning(f"Could not detect embedding dimension, using default {embedding_dim}: {str(e)}")

    try:
        collections = client.get_collections().collections
        collection_exists = any(c.name == collection_name for c in collections)

        if not collection_exists:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {collection_name}")
        else:
            logger.info(f"Using existing Qdrant collection: {collection_name}")

        for field_name in ("user_id", "task_id"):
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                logger.info(f"Ensured Qdrant payload index: {collection_name}.{field_name}")
            except Exception as e:
                logger.warning(f"Could not create/ensure index for {collection_name}.{field_name}: {e}")
    except Exception as e:
        logger.warning(f"Collection {collection_name} operation issue: {str(e)}")

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        vectorstore.add_documents(batch)

    logger.info(f"Added {len(documents)} documents to Qdrant collection: {collection_name}")
    return vectorstore


def query_vectorstore_multi_task(
    question: str,
    user_id: str,
    task_ids: List[str],
    k: int = 4,
) -> List[Document]:
    """Query multiple Qdrant collections (tasks) simultaneously."""
    client = get_qdrant_client()
    embeddings = get_embedding_model()

    all_documents = []
    for task_id in task_ids:
        collection_name = get_collection_name(user_id, task_id)
        try:
            collections = client.get_collections().collections
            collection_exists = any(c.name == collection_name for c in collections)
            if not collection_exists:
                logger.warning(f"Collection {collection_name} does not exist, skipping")
                continue

            vectorstore = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embeddings,
            )

            docs = vectorstore.similarity_search(
                question,
                k=k,
            )
            all_documents.extend(docs)
            logger.info(f"Retrieved {len(docs)} documents from task_id: {task_id}")
        except Exception as e:
            logger.error(f"Error querying collection {collection_name}: {str(e)}")
            continue

    return all_documents


def get_task_collections(user_id: str) -> List[Dict[str, Any]]:
    """Get list of collections (tasks) for a user."""
    client = get_qdrant_client()
    safe_user_id = user_id.replace("-", "_").replace("/", "_")
    prefix = f"{safe_user_id}_"

    try:
        collections = client.get_collections().collections
        user_collections = []

        for collection in collections:
            if collection.name.startswith(prefix):
                task_id_part = collection.name[len(prefix) :]
                task_id = task_id_part.replace("_", "-")

                try:
                    info = client.get_collection(collection.name)
                    user_collections.append({
                        "task_id": task_id,
                        "collection_name": collection.name,
                        "points_count": info.points_count,
                        "vectors_count": info.vectors_count,
                    })
                except Exception as e:
                    logger.warning(f"Could not get info for collection {collection.name}: {str(e)}")
                    user_collections.append({
                        "task_id": task_id,
                        "collection_name": collection.name,
                        "points_count": 0,
                        "vectors_count": 0,
                    })

        return user_collections
    except Exception as e:
        logger.error(f"Error retrieving Qdrant collections: {str(e)}")
        return []
