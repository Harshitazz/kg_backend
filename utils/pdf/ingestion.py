import logging
import os
import traceback
import uuid
from typing import Dict, List, Optional

from fastapi import BackgroundTasks, HTTPException, UploadFile
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from cache import cache_query, get_cached_query
from repositories.qdrant_repository import create_vectorstore_for_task, query_vectorstore_multi_task
from repositories.storage_repository import download_file_from_storage, list_storage_objects, upload_file_to_storage
from kg_backend.utils.kg.storage import process_documents_for_kg
from kg_backend.utils.pdf.common import AskPDFRequest, DEFAULT_PROMPT_TEMPLATE, DEFAULT_SEARCH_K, EMBEDDING_MODEL_NAME, get_embedding_model, llm
from kg_backend.utils.pdf.ocr import OCR_AVAILABLE, extract_text_with_ocr_fallback

logger = logging.getLogger(__name__)

# Keep llm in sync with the PDF service public API.

def set_llm(llm_instance):
    global llm
    llm = llm_instance


def process_pdf_documents(task_id: str, user_id: str, task_status: Dict):
    """Downloads PDFs, generates vector embeddings, and stores in Qdrant."""
    try:
        task_status[task_id] = "Processing"
        pdf_prefix = f"pdf_uploads/{user_id}/{task_id}/"
        objects = list_storage_objects(pdf_prefix)

        all_docs: List[Document] = []
        failed_pdfs = []
        if not objects:
            raise Exception("No PDFs found in storage for processing.")

        for obj in objects:
            pdf_key = obj["Key"]
            local_pdf_path = f"/tmp/{os.path.basename(pdf_key)}"

            print(f"Downloading: {pdf_key}")
            os.makedirs("/tmp", exist_ok=True)
            download_file_from_storage(pdf_key, local_pdf_path)

            pdf_docs = []
            skipped_pages = []
            needs_ocr = False
            ocr_error = None
            total_pages = 0

            with __import__("fitz").open(local_pdf_path) as doc:
                total_pages = len(doc)
                for page_num, page in enumerate(doc):
                    try:
                        text = extract_text_with_ocr_fallback(page, pdf_key, page_num)

                        if not page.get_text("text").strip() and text.strip():
                            needs_ocr = True

                        if text.strip():
                            pdf_docs.append(
                                Document(
                                    page_content=text,
                                    metadata={
                                        "page": page_num + 1,
                                        "source": pdf_key,
                                        "total_pages": total_pages,
                                    },
                                )
                            )
                        else:
                            if not OCR_AVAILABLE:
                                skipped_pages.append(page_num + 1)
                                print(f"Skipping page {page_num + 1} in {pdf_key}: No text and OCR not available")
                            else:
                                skipped_pages.append(page_num + 1)
                                print(f"Skipping page {page_num + 1} in {pdf_key}: No extractable text (blank or poor quality image)")
                    except Exception as exc:
                        error_msg = str(exc)
                        if "OCR configuration error" in error_msg or "Tesseract" in error_msg:
                            ocr_error = error_msg
                            skipped_pages.append(page_num + 1)
                            print(f"OCR error on page {page_num + 1} in {pdf_key}: {error_msg}")
                        else:
                            skipped_pages.append(page_num + 1)
                            print(f"Error processing page {page_num + 1} in {pdf_key}: {error_msg}")

            filename = os.path.basename(pdf_key)
            if not pdf_docs:
                if not OCR_AVAILABLE:
                    error_msg = f"{filename}: Image-based PDF requires OCR, but Tesseract is not installed. Skipped."
                elif ocr_error and ("Tesseract" in ocr_error or "not installed" in ocr_error.lower()):
                    error_msg = f"{filename}: OCR configuration error - {ocr_error}. Skipped."
                elif needs_ocr:
                    error_msg = f"{filename}: Image-based PDF could not be processed with OCR (all pages failed). Skipped."
                else:
                    error_msg = f"{filename}: No extractable text found. Skipped."
                failed_pdfs.append(error_msg)
                print(f"Warning: {error_msg}")
            else:
                success_msg = f"{filename}: Processed {len(pdf_docs)}/{total_pages} pages"
                if skipped_pages:
                    success_msg += f" (skipped pages: {', '.join(map(str, skipped_pages))})"
                if needs_ocr:
                    success_msg += " [used OCR]"
                print(f"Success: {success_msg}")

            if os.path.exists(local_pdf_path):
                os.remove(local_pdf_path)

            if pdf_docs:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=3500,
                    chunk_overlap=200,
                )
                split_docs = text_splitter.split_documents(pdf_docs)
                all_docs.extend(split_docs)
                print(f"Added {len(split_docs)} document chunks from {filename}")

                try:
                    print(f"Creating knowledge graph for {filename} (task_id: {task_id})...")
                    process_documents_for_kg(split_docs, user_id, source=filename, task_id=task_id)
                    print(f"Knowledge graph created for {filename} (task_id: {task_id})")
                except Exception as kg_error:
                    print(f"Warning: Failed to create knowledge graph for {filename}: {str(kg_error)}")

        if not all_docs:
            error_details = "\n".join(failed_pdfs) if failed_pdfs else "Unknown error"
            error_message = "No valid text found in any PDFs. Vector index cannot be created.\n\nFailed PDFs:\n" + error_details
            if not OCR_AVAILABLE and any("Tesseract" in msg for msg in failed_pdfs):
                error_message += "\n\nNote: Some PDFs appear to be image-based and require Tesseract OCR. Install Tesseract to process image-based PDFs."
            raise Exception(error_message)

        successful_count = len(objects) - len(failed_pdfs)
        if failed_pdfs:
            print(f"\nProcessing summary: {successful_count} PDF(s) processed successfully, {len(failed_pdfs)} PDF(s) skipped")
            print(f"Skipped PDFs: {', '.join([os.path.basename(msg.split(':')[0]) for msg in failed_pdfs])}")
        else:
            print(f"\nProcessing summary: All {len(objects)} PDF(s) processed successfully")

        print(f"Generating vector embeddings and storing in Qdrant from {len(all_docs)} document chunks...")
        try:
            source_groups = {}
            for doc in all_docs:
                source = doc.metadata.get("source", "unknown")
                source_groups.setdefault(source, []).append(doc)

            create_vectorstore_for_task(
                documents=all_docs,
                user_id=user_id,
                task_id=task_id,
                source=",".join(source_groups.keys()),
            )
            print(f"✓ Successfully stored {len(all_docs)} documents in Qdrant for task_id: {task_id}")
        except Exception as qdrant_error:
            print(f"✗ Qdrant storage failed: {str(qdrant_error)}")
            raise Exception(f"Failed to store documents in Qdrant: {str(qdrant_error)}. Please ensure Qdrant is running.")

        task_status[task_id] = "Completed"

    except Exception as exc:
        print(f"Error in process_pdf_documents: {str(exc)}")
        task_status[task_id] = f"Failed: {str(exc)}"
        raise


async def upload_pdfs_endpoint(background_tasks: BackgroundTasks, files: list[UploadFile], task_status: Dict, request_user: Dict[str, str], user_tasks: Dict[str, list]):
    try:
        user_id = request_user["user_id"]
        s3_urls = []
        task_id = str(uuid.uuid4())

        for file in files:
            if not file.filename or not file.filename.endswith(".pdf"):
                raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

            local_pdf_path = f"uploads/{user_id}_{file.filename}"
            os.makedirs("uploads", exist_ok=True)

            with open(local_pdf_path, "wb") as f:
                f.write(await file.read())

            pdf_key = f"pdf_uploads/{user_id}/{task_id}/{file.filename}"
            storage_url = upload_file_to_storage(local_pdf_path, pdf_key)
            os.remove(local_pdf_path)
            s3_urls.append(storage_url)

        task_status[task_id] = "Pending"
        if user_id not in user_tasks:
            user_tasks[user_id] = []
        user_tasks[user_id].append(task_id)

        background_tasks.add_task(process_pdf_documents, task_id, user_id, task_status)
        return {"message": "PDF processing started", "task_id": task_id, "pdf_s3_urls": s3_urls}

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=f"Configuration error: {str(exc)}")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"File error: {str(exc)}")
    except Exception as exc:
        print(f"Error in upload_pdfs_endpoint: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(exc)}")


async def ask_pdf_endpoint(request: AskPDFRequest, request_user: Dict[str, str], task_status: Dict, user_tasks: Dict[str, list]):
    """Ask a question based on PDF data using Qdrant and KG context."""
    user_id = request_user["user_id"]
    question = request.question or ""
    task_ids = request.task_ids or []

    if not task_ids:
        task_id = request.task_id
        task_ids = [task_id] if task_id else []

    search_params = {"k": DEFAULT_SEARCH_K}
    prompt_version = DEFAULT_PROMPT_TEMPLATE

    valid_task_ids = []
    for task_id in task_ids:
        if task_id and len(task_id) > 20 and "-" in task_id:
            valid_task_ids.append(task_id)
        else:
            logger.warning(f"Invalid task_id format, skipping: {task_id}")

    if not valid_task_ids:
        raise HTTPException(status_code=400, detail="No valid task_ids provided. Task IDs must be valid UUIDs.")

    cached_result = get_cached_query(
        question,
        user_id=user_id,
        task_ids=valid_task_ids,
        embedding_model=EMBEDDING_MODEL_NAME,
        search_params=search_params,
        prompt_version=prompt_version,
    )
    if cached_result:
        return cached_result

    try:
        all_documents = query_vectorstore_multi_task(
            question=question,
            user_id=user_id,
            task_ids=valid_task_ids,
            k=search_params["k"],
        )
        logger.info(f"Retrieved {len(all_documents)} documents from Qdrant for {len(valid_task_ids)} tasks")
    except Exception as qdrant_error:
        logger.error(f"Qdrant query failed: {str(qdrant_error)}")
        raise HTTPException(status_code=503, detail=f"Vector database unavailable: {str(qdrant_error)}. Please ensure Qdrant is running.")

    from kg_backend.utils.kg.query import query_kg_for_question

    kg_context = query_kg_for_question(question, user_id, task_ids=valid_task_ids if valid_task_ids else None)

    if not question:
        raise HTTPException(status_code=400, detail="No question provided.")

    unique_docs = []
    seen_content = set()
    if all_documents:
        for doc in all_documents:
            content_hash = hash(doc.page_content[:200])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)

        max_docs = search_params["k"] * max(len(task_ids), 1) if task_ids else search_params["k"]
        unique_docs = unique_docs[:max_docs]

        def format_document_source(source: str) -> str:
            if not source or source == "unknown":
                return "document"
            return os.path.basename(source)

        def format_context(docs):
            formatted_parts = []
            for i, doc in enumerate(docs, 1):
                source = format_document_source(doc.metadata.get("source", "unknown"))
                page = doc.metadata.get("page", "?")
                total_pages = doc.metadata.get("total_pages", "?")

                if page != "?":
                    source_ref = f"{source} (page {page}"
                    if total_pages != "?":
                        source_ref += f" of {total_pages}"
                    source_ref += ")"
                else:
                    source_ref = source

                formatted_parts.append(f"[From {source_ref}]\n{doc.page_content}")

            return "\n\n---\n\n".join(formatted_parts)

        formatted_context = format_context(unique_docs)
        if kg_context:
            formatted_context = formatted_context + "\n\n" + kg_context
    else:
        if kg_context:
            formatted_context = kg_context
        else:
            if task_ids:
                raise HTTPException(status_code=404, detail=f"No relevant content found in selected tasks: {', '.join(task_ids)}")
            raise HTTPException(status_code=404, detail="No relevant content found. Please upload documents first.")

    enhanced_prompt = prompt_version + "\n\nNote: When referencing sources in your answer, use the document filename and page number (e.g., 'resume.pdf, page 2'). Do not reference document IDs."
    prompt = PromptTemplate.from_template(enhanced_prompt)
    chain = prompt | llm | StrOutputParser()

    logger.info("=" * 80)
    logger.info("QUESTION: %s", question)
    logger.info("QDRANT DOC COUNT: %d", len(unique_docs))
    logger.info("KG CONTEXT LENGTH: %d", len(kg_context or ""))
    logger.info("FORMATTED CONTEXT LENGTH: %d", len(formatted_context))
    logger.info("FORMATTED CONTEXT PREVIEW:\n%s", formatted_context[:12000])
    logger.info("=" * 80)
    answer = chain.invoke({"context": formatted_context, "question": question})

    relevant_nodes = []
    if kg_context:
        from kg_backend.utils.kg.query import get_relevant_nodes_for_question

        try:
            relevant_nodes = get_relevant_nodes_for_question(question, user_id, task_ids=valid_task_ids if valid_task_ids else None)
        except Exception as exc:
            logger.warning(f"Failed to get relevant nodes: {str(exc)}")

    source_chunks = []
    if all_documents:
        for doc in unique_docs:
            source = os.path.basename(doc.metadata.get("source", "unknown"))
            page = doc.metadata.get("page", "?")
            total_pages = doc.metadata.get("total_pages", "?")

            source_chunks.append({
                "text": doc.page_content,
                "source": source,
                "page": page,
                "total_pages": total_pages,
                "metadata": doc.metadata,
            })

    response = {
        "answer": answer,
        "source_chunks": source_chunks,
        "relevant_nodes": relevant_nodes,
    }

    cache_query(
        question,
        {"answer": answer},
        user_id=user_id,
        task_ids=valid_task_ids if valid_task_ids else None,
        embedding_model=EMBEDDING_MODEL_NAME,
        search_params=search_params,
        prompt_version=prompt_version,
    )
    return response
