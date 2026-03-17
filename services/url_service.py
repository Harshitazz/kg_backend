from fastapi import BackgroundTasks, HTTPException, Request
from pydantic import BaseModel
import os
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import Dict
from cache import get_cached_query, cache_query
from langchain_groq import ChatGroq
from services.kg_service import process_documents_for_kg
from services.qdrant_service import create_vectorstore_for_task

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SEARCH_K = 4
DEFAULT_PROMPT_TEMPLATE = """Use the following context to answer the question.
    Rules:
    - Do NOT mention document IDs, UUIDs, or internal references.
    - Do NOT mention phrases like "document with id".
    - Refer to documents only by their source name or page number.
    - If the answer can be inferred by combining information from multiple documents, do so.
    - If the information is truly missing, say "I don't know".

    Context:
    {context}

    Question: {question}

    Answer:"""

llm = None

def set_llm(llm_instance):
    """Set the LLM instance from main.py"""
    global llm
    llm = llm_instance

async def process_url_documents(task_id: str, urls: list[str], task_status: Dict, file_path: str, user_id: str = "default"):
    try:
        task_status[task_id] = "Processing"
        print("task1")
        # Load data
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()
        print("task2")
        # Split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=3500  # Increased from 1000 to create larger chunks
        )
        docs = text_splitter.split_documents(data)
        print("task3")
        
        # Create knowledge graph for URLs
        try:
            print(f"Creating knowledge graph for URLs (task_id: {task_id})...")
            for i, url in enumerate(urls):
                # Process documents that came from this URL
                url_docs = [doc for doc in docs if doc.metadata.get("source", "").startswith(url)]
                if url_docs:
                    process_documents_for_kg(url_docs, user_id, source=url, task_id=task_id)
            print(f"Knowledge graph created for URLs (task_id: {task_id})")
        except Exception as kg_error:
            print(f"Warning: Failed to create knowledge graph for URLs: {str(kg_error)}")
            # Continue processing even if KG creation fails
        
        # Create embeddings and store in Qdrant
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("task4")
        
        # Use Qdrant for persistent vector storage
        try:
            # Group documents by URL source
            url_sources = [url for url in urls]
            source_str = ",".join(url_sources[:3])  # Limit source string length
            
            vectorstore = create_vectorstore_for_task(
                documents=docs,
                user_id=user_id,
                task_id=task_id,
                source=source_str
            )
            print(f"✓ Successfully stored {len(docs)} documents in Qdrant for task_id: {task_id}")
            
        except Exception as qdrant_error:
            print(f"✗ Qdrant storage failed: {str(qdrant_error)}")
            raise Exception(f"Failed to store documents in Qdrant: {str(qdrant_error)}. Please ensure Qdrant is running.")
        
        print("Vector index initialized successfully.")
        task_status[task_id] = "Completed"
    except Exception as e:
        task_status[task_id] = f"Failed: {str(e)}"


async def initialize_vector_index_endpoint(background_tasks: BackgroundTasks, request: Request, task_status: Dict, file_path: str, user_id: str = "default"):
    try:
        data = await request.json()  # Ensure JSON is properly parsed
        urls = data.get("urls", [])

        if not urls:
            raise HTTPException(status_code=400, detail="No URLs provided.")

        task_id = str(uuid.uuid4())
        task_status[task_id] = "Pending"
        background_tasks.add_task(process_url_documents, task_id, urls, task_status, file_path, user_id)

        return {"message": "Vector index initialization started in the background.", "task_id": task_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def ask_endpoint(request: Request, file_path: str):
    """Legacy endpoint - use /ask_pdf instead which uses Qdrant"""
    data = await request.json()
    question = data.get("question", "")
    task_id = data.get("task_id")
    
    if not question:
        raise HTTPException(status_code=400, detail="No question provided.")
    
    # This endpoint is deprecated - redirect to use Qdrant-based endpoint
    raise HTTPException(
        status_code=410,
        detail="This endpoint is deprecated. Please use /ask_pdf endpoint with task_ids parameter for Qdrant-based querying."
    )
