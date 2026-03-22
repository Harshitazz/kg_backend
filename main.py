from fastapi import FastAPI, BackgroundTasks, UploadFile, Depends, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict
import os
import logging
import dotenv
dotenv.load_dotenv()

from langchain_groq import ChatGroq
from auth import get_current_user
from config import get_settings
from services.pdf_service import (
    upload_pdfs_endpoint,
    ask_pdf_endpoint,
    set_llm as set_pdf_llm,
    AskPDFRequest,
)
from services.url_service import (
    initialize_vector_index_endpoint,
    ask_endpoint,
    set_llm as set_url_llm,
)
from services.kg_service import set_llm as set_kg_llm, get_user_knowledge_graph, get_user_task_ids, get_knowledge_graph_history, get_node_explanation
from storage import list_storage_objects

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info("Starting FastAPI application...")

# Load and validate settings
try:
    settings = get_settings()
    logging.info(f"Settings loaded successfully. Environment: {settings.environment}")
except Exception as e:
    logging.error(f"Failed to load settings: {e}")
    settings = None

task_status: Dict[str, str] = {} 
user_tasks: Dict[str, list] = {} 
file_path = "storage/vector_index"  # Legacy path, not used with Qdrant

app = FastAPI(redirect_slashes=False)

# CORS configuration - restrict to specific origins for security
if settings:
    allowed_origins_str = settings.allowed_origins
    allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()]
else:
    allowed_origins = []

# If no origins specified, default to common development origins
if not allowed_origins:
    allowed_origins = [
        "http://localhost:3000",
        "http://localhost:3001",
        "https://kq-frontend-one.vercel.app",
    ]

logging.info(f"CORS allowed origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    
    auth_header = request.headers.get("authorization", "No auth header")
    if auth_header.startswith("Bearer "):
        logging.info(f"Auth header present: Bearer {auth_header[7:27]}...")
    else:
        logging.info(f"Auth header: {auth_header}")
    
    try:
        response = await call_next(request)
        logging.info(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logging.error(f"Request failed: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {str(e)}"}
        )

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    if not settings:
        logging.error("Settings not loaded — skipping initialization")
        return
    
    if not settings.groq_api_key:
        logging.error("GROQ_API_KEY missing — LLM not initialized")
        return

    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=settings.groq_api_key,
            temperature=0.9,
            max_tokens=1024,
        )
        set_pdf_llm(llm)
        set_url_llm(llm)
        set_kg_llm(llm)
        logging.info("LLM initialized successfully")
    except Exception as e:
        logging.exception(f"LLM initialization failed: {e}")

@app.get("/")
def health():
    logging.info("Health check endpoint called")
    return {"status": "ok", "message": "VectraMind API is running"}

@app.post("/initialize_vector_index")
async def initialize_vector_index(background_tasks: BackgroundTasks, request: Request, request_user=Depends(get_current_user)):
    logging.info("Initialize vector index endpoint called (Qdrant)")
    user_id = request_user.get("user_id", "default")
    return await initialize_vector_index_endpoint(
        background_tasks, request, task_status, file_path, user_id
    )

@app.get("/task_status/{task_id}")
def get_task_status(task_id: str):
    status = task_status.get(task_id, "Not found")
    logging.info(f"Task status check: {task_id} = {status}")
    return {"task_id": task_id, "status": status}

@app.post("/ask")
async def ask(request: Request):
    logging.info("Ask endpoint called")
    return await ask_endpoint(request, file_path)

@app.post("/upload_pdfs")
@app.post("/upload_pdfs/") 
async def upload_pdfs(
    background_tasks: BackgroundTasks,
    files: list[UploadFile],
    request_user=Depends(get_current_user),
):
    logging.info(f"Upload PDFs endpoint called by user: {request_user.get('user_id')}")
    logging.info(f"Received {len(files)} file(s)")
    return await upload_pdfs_endpoint(
        background_tasks, files, task_status, request_user, user_tasks
    )

@app.post("/ask_pdf")
async def ask_pdf(
    request: AskPDFRequest,
    request_user=Depends(get_current_user),
):
    logging.info(f"Ask PDF endpoint called by user: {request_user.get('user_id')}")
    logging.info(f"Question: {request.question}")
    
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
                "task_id": active_tasks[0],  # Primary task ID
                "task_ids": active_tasks,  # All active task IDs
                "status_url": f"/task_status/{active_tasks[0]}"
            }
        )
    
    return await ask_pdf_endpoint(request, request_user, task_status, user_tasks)

@app.get("/knowledge_graph")
async def get_knowledge_graph(
    request_user=Depends(get_current_user),
    task_id: str = None,
    limit: int = 50
):
    """Get knowledge graph data for the current user, optionally filtered by task_id"""
    logging.info(f"Knowledge graph endpoint called by user: {request_user.get('user_id')}, task_id: {task_id}")
    user_id = request_user.get("user_id")
    
    try:
        graph_data = get_user_knowledge_graph(user_id, task_id=task_id, limit=limit)
        return graph_data
    except Exception as e:
        logging.error(f"Error retrieving knowledge graph: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve knowledge graph: {str(e)}"
        )

@app.get("/knowledge_graph/tasks")
async def get_knowledge_graph_tasks(
    request_user=Depends(get_current_user)
):
    """Get list of task_ids that have knowledge graphs for the current user"""
    logging.info(f"Knowledge graph tasks endpoint called by user: {request_user.get('user_id')}")
    user_id = request_user.get("user_id")
    
    try:
        task_ids = get_user_task_ids(user_id)
        return {"task_ids": task_ids}
    except Exception as e:
        logging.error(f"Error retrieving task IDs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve task IDs: {str(e)}"
        )

@app.get("/knowledge_graph/history")
async def get_knowledge_graph_history_endpoint(
    request_user=Depends(get_current_user)
):
    """Get history of knowledge graphs for the current user"""
    logging.info(f"Knowledge graph history endpoint called by user: {request_user.get('user_id')}")
    user_id = request_user.get("user_id")
    
    try:
        history = get_knowledge_graph_history(user_id)
        return {"history": history}
    except Exception as e:
        logging.error(f"Error retrieving KG history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve KG history: {str(e)}"
        )

@app.get("/knowledge_graph/node/explain")
async def explain_node(
    node_name: str,
    task_id: str = None,
    request_user=Depends(get_current_user)
):
    """Get explanation for a specific node"""
    logging.info(f"Node explanation requested for: {node_name}, user: {request_user.get('user_id')}")
    user_id = request_user.get("user_id")
    
    try:
        explanation = get_node_explanation(node_name, user_id, task_id=task_id)
        return {"node_name": node_name, "explanation": explanation}
    except Exception as e:
        logging.error(f"Error generating node explanation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate explanation: {str(e)}"
        )

@app.get("/vector/history")
async def get_vector_history(
    request_user=Depends(get_current_user)
):
    """Get history of vector indexes (Qdrant collections) for the current user"""
    logging.info(f"Vector index history endpoint called by user: {request_user.get('user_id')}")
    user_id = request_user.get("user_id")
    
    try:
        # Try Qdrant first (preferred)
        try:
            from services.qdrant_service import get_task_collections
            qdrant_collections = get_task_collections(user_id)
            
            if qdrant_collections:
                history = []
                for coll in qdrant_collections:
                    history.append({
                        "key": f"{user_id}/qdrant/{coll['task_id']}",  # For compatibility
                        "task_id": coll["task_id"],
                        "created_at": "",  # Qdrant doesn't store creation time
                        "size": coll.get("points_count", 0)  # Number of vectors
                    })
                
                return {"history": history}
        except Exception as qdrant_error:
            logging.error(f"Qdrant query failed: {str(qdrant_error)}")
        
        # If Qdrant fails, return empty history
        logging.error("Qdrant is required for vector storage. Please ensure Qdrant is running.")
        return {"history": []}
    except Exception as e:
        logging.error(f"Error retrieving vector index history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve vector index history: {str(e)}"
        )

@app.post("/vector/query/{task_id}")
async def query_task_vector(
    task_id: str,
    question: str,
    request_user=Depends(get_current_user)
):
    """Query a specific task's vector embeddings (Qdrant)"""
    from services.pdf_service import ask_pdf_endpoint, AskPDFRequest
    
    user_id = request_user.get("user_id")
    request_data = AskPDFRequest(question=question, task_ids=[task_id])
    
    try:
        result = await ask_pdf_endpoint(request_data, request_user, task_status={}, user_tasks={})
        return result
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Vector index not found for task_id: {task_id}. Error: {str(e)}"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled exception: {str(exc)}")
    import traceback
    logging.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"Internal server error: {str(exc)}",
            "type": type(exc).__name__
        }
    )