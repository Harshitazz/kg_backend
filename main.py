from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import dotenv

dotenv.load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from config import get_settings
from controllers.health_controller import router as health_router
from controllers.pdf_controller import router as pdf_router
from controllers.url_controller import router as url_router
from controllers.knowledge_graph_controller import router as knowledge_graph_router
from controllers.vector_controller import router as vector_router
from services.pdf_service import set_llm as set_pdf_llm
from services.url_service import set_llm as set_url_llm
from services.kg_service import set_llm as set_kg_llm

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

app = FastAPI(redirect_slashes=False)

app.include_router(health_router)
app.include_router(url_router)
app.include_router(pdf_router)
app.include_router(knowledge_graph_router)
app.include_router(vector_router)

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
    
    if not settings.gemini_api_key:
        logging.error("GEMINI_API_KEY missing — LLM not initialized")
        return

    try:
        kg_llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0,
            response_mime_type="application/json",
        )
        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0.6,
        )
        set_pdf_llm(llm)
        set_url_llm(llm)
        set_kg_llm(kg_llm)
        logging.info("LLM initialized successfully")
    except Exception as e:
        logging.exception(f"LLM initialization failed: {e}")

