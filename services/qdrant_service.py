"""
Qdrant Vector Store Service for persistent vector storage
"""
import os
import logging
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Qdrant connection settings
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

_qdrant_client = None
_embedding_model = None

def get_qdrant_client() -> QdrantClient:
    """Get or create Qdrant client"""
    global _qdrant_client
    if _qdrant_client is None:
        try:
            _qdrant_client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
            )
            # Test connection
            _qdrant_client.get_collections()
            logger.info(f"Connected to Qdrant at {QDRANT_URL}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            raise
    return _qdrant_client

def get_embedding_model():
    """Get embedding model (shared with pdf_service)"""
    global _embedding_model
    if _embedding_model is None:
        from services.pdf_service import get_embedding_model
        _embedding_model = get_embedding_model()
    return _embedding_model

def get_collection_name(user_id: str, task_id: str) -> str:
    """Generate collection name for user and task"""
    # Qdrant collection names must be valid identifiers
    # Replace special characters that might cause issues
    safe_user_id = user_id.replace("-", "_").replace("/", "_")
    safe_task_id = task_id.replace("-", "_").replace("/", "_")
    return f"{safe_user_id}_{safe_task_id}"

def create_vectorstore_for_task(
    documents: List[Document],
    user_id: str,
    task_id: str,
    source: str
) -> QdrantVectorStore:
    """
    Create Qdrant vectorstore for a specific task
    
    Args:
        documents: List of Document objects with text and metadata
        user_id: User ID
        task_id: Task ID
        source: Source identifier (PDF filename or URL)
    
    Returns:
        QdrantVectorStore instance
    """
    client = get_qdrant_client()
    embeddings = get_embedding_model()
    collection_name = get_collection_name(user_id, task_id)
    
    # Add metadata to all documents
    for doc in documents:
        if not doc.metadata:
            doc.metadata = {}
        doc.metadata["user_id"] = user_id
        doc.metadata["task_id"] = task_id
        if "source" not in doc.metadata:
            doc.metadata["source"] = source
    
    # Get embedding dimension
    try:
        test_embedding = embeddings.embed_query("test")
        embedding_dim = len(test_embedding)
        logger.debug(f"Detected embedding dimension: {embedding_dim}")
    except Exception as e:
        # Fallback to common dimension for all-MiniLM-L6-v2
        embedding_dim = 384
        logger.warning(f"Could not detect embedding dimension, using default {embedding_dim}: {str(e)}")
    
    # Create or get collection
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_exists = any(c.name == collection_name for c in collections)
        
        if not collection_exists:
            # Create collection
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created Qdrant collection: {collection_name}")
        else:
            logger.info(f"Using existing Qdrant collection: {collection_name}")
    except Exception as e:
        logger.warning(f"Collection {collection_name} operation issue: {str(e)}")
        # Try to continue anyway
    
    # Create vectorstore and add documents
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    
    # Add documents in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        vectorstore.add_documents(batch)
    
    logger.info(f"Added {len(documents)} documents to Qdrant collection: {collection_name}")
    
    return vectorstore

def query_vectorstore_multi_task(
    question: str,
    user_id: str,
    task_ids: List[str],
    k: int = 4
) -> List[Document]:
    """
    Query multiple Qdrant collections (tasks) simultaneously
    
    Args:
        question: Query question
        user_id: User ID
        task_ids: List of task IDs to query
        k: Number of results per task
    
    Returns:
        List of relevant documents from all tasks
    """
    client = get_qdrant_client()
    embeddings = get_embedding_model()
    
    all_documents = []
    
    for task_id in task_ids:
        collection_name = get_collection_name(user_id, task_id)
        
        try:
            # Check if collection exists
            collections = client.get_collections().collections
            collection_exists = any(c.name == collection_name for c in collections)
            
            if not collection_exists:
                logger.warning(f"Collection {collection_name} does not exist, skipping")
                continue
            
            # Query this collection
            vectorstore = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embeddings,
            )
            
            # Create filter for user_id and task_id
            qdrant_filter = Filter(
                must=[
                    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                    FieldCondition(key="task_id", match=MatchValue(value=task_id))
                ]
            )
            
            # Retrieve documents
            docs = vectorstore.similarity_search(
                question,
                k=k,
                filter=qdrant_filter
            )
            
            all_documents.extend(docs)
            logger.info(f"Retrieved {len(docs)} documents from task_id: {task_id}")
            
        except Exception as e:
            logger.error(f"Error querying collection {collection_name}: {str(e)}")
            continue
    
    return all_documents

def get_task_collections(user_id: str) -> List[Dict[str, Any]]:
    """
    Get list of collections (tasks) for a user
    
    Returns:
        List of dicts with task_id, collection_name, document_count
    """
    client = get_qdrant_client()
    safe_user_id = user_id.replace("-", "_").replace("/", "_")
    prefix = f"{safe_user_id}_"
    
    try:
        collections = client.get_collections().collections
        user_collections = []
        
        for collection in collections:
            if collection.name.startswith(prefix):
                # Extract task_id from collection name
                task_id_part = collection.name[len(prefix):]
                # Convert back from safe format
                task_id = task_id_part.replace("_", "-")
                
                # Get collection info
                try:
                    info = client.get_collection(collection.name)
                    user_collections.append({
                        "task_id": task_id,
                        "collection_name": collection.name,
                        "points_count": info.points_count,
                        "vectors_count": info.vectors_count
                    })
                except Exception as e:
                    logger.warning(f"Could not get info for collection {collection.name}: {str(e)}")
                    user_collections.append({
                        "task_id": task_id,
                        "collection_name": collection.name,
                        "points_count": 0,
                        "vectors_count": 0
                    })
        
        return user_collections
    except Exception as e:
        logger.error(f"Error getting user collections: {str(e)}")
        return []

def delete_collection(user_id: str, task_id: str) -> bool:
    """
    Delete a Qdrant collection for a task
    
    Args:
        user_id: User ID
        task_id: Task ID
    
    Returns:
        True if deleted, False otherwise
    """
    try:
        client = get_qdrant_client()
        collection_name = get_collection_name(user_id, task_id)
        client.delete_collection(collection_name)
        logger.info(f"Deleted Qdrant collection: {collection_name}")
        return True
    except Exception as e:
        logger.error(f"Error deleting collection {collection_name}: {str(e)}")
        return False
