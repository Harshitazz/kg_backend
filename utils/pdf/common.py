import logging
from typing import List, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SEARCH_K = 8
DEFAULT_PROMPT_TEMPLATE = """You are the answer-generation component of a Knowledge Graph and semantic document retrieval system.

Answer the user's question using BOTH:
1. The retrieved document chunks.
2. The knowledge graph context.

Important:
- The answer may require combining information from multiple document chunks.
- The answer may require connecting concepts through relationships in the knowledge graph.
- Do not require the exact wording of the question to appear in a document.
- If multiple pieces of context together support the answer, synthesize them.
- Do not invent facts that are not supported by the provided context.
- Give a concise but complete answer.
- When useful, explain the relationship between concepts step-by-step.
- Mention the relevant document filename when useful.
- Do not mention internal IDs, UUIDs, task IDs, or database implementation details.

Only say:
"I couldn't find this information in the provided documents."
if the provided document chunks AND knowledge graph context genuinely contain no information that can answer the question.

Context:
{context}

Question:
{question}

Answer:"""

_embedding_model = None
llm = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embedding_model


def set_llm(llm_instance):
    global llm
    llm = llm_instance


class AskPDFRequest(BaseModel):
    question: str
    task_id: Optional[str] = None
    task_ids: Optional[List[str]] = None
