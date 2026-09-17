import os
import shutil
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body
from pydantic import BaseModel
from document_loader import (
    load_and_split_pdf,
    load_and_split_multiple_pdfs,
    load_and_split_directory
)
from vector_store import add_documents_to_vectorstore, get_indexed_documents
from rag_chain import ask_question, summarize_documents

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(
    title="RAG & Document Summarization API",
    description="Scalable API for multi-PDF ingestion (100+ PDFs), vector search Q&A, and structured summarization.",
    version="2.0.0"
)

# Mount static directory
os.makedirs("./static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=FileResponse)
def read_root():
    """Serve the Web UI SPA."""
    return FileResponse("static/index.html")


# Schemas
class QueryRequest(BaseModel):
    question: str
    filename_filter: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    context: List[str]
    sources: List[str]

class SummarizeRequest(BaseModel):
    topic_or_query: Optional[str] = None
    filename_filter: Optional[str] = None

class SummarizeResponse(BaseModel):
    summary: str
    sources_used: List[str]
    chunks_analyzed: int

class IngestResponse(BaseModel):
    files_processed: List[str]
    chunks_processed: int
    message: str

class DirectoryIngestRequest(BaseModel):
    directory_path: str

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "message": "RAG & Summarization API is running."}

@app.get("/api/v1/documents")
def list_documents():
    """List all ingested document filenames stored in vector store."""
    docs = get_indexed_documents()
    return {"total_documents": len(docs), "documents": docs}

@app.post("/api/v1/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """Ask a question to the RAG pipeline with optional filename filtering."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        response = ask_question(request.question, filename_filter=request.filename_filter)
        return QueryResponse(
            answer=response["answer"],
            context=response["context"],
            sources=response["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing RAG query: {str(e)}")

@app.post("/api/v1/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest):
    """Generate structured summary for a specific document or across the entire PDF collection."""
    try:
        response = summarize_documents(
            topic_or_query=request.topic_or_query,
            filename_filter=request.filename_filter
        )
        return SummarizeResponse(
            summary=response["summary"],
            sources_used=response["sources_used"],
            chunks_analyzed=response["chunks_analyzed"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

@app.post("/api/v1/ingest", response_model=IngestResponse)
async def ingest_single_document(file: UploadFile = File(...)):
    """Upload a single PDF file and store into Chroma vector store."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    temp_dir = "./temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        chunks = load_and_split_pdf(temp_file_path)
        add_documents_to_vectorstore(chunks)
        
        return IngestResponse(
            files_processed=[file.filename],
            chunks_processed=len(chunks),
            message="Document successfully ingested into vector store."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting document: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/api/v1/ingest/batch", response_model=IngestResponse)
async def ingest_batch_documents(files: List[UploadFile] = File(...)):
    """Upload multiple PDF files at once and ingest them into vector store."""
    temp_dir = "./temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    saved_paths = []
    
    try:
        for file in files:
            if not file.filename.endswith(".pdf"):
                continue
            path = os.path.join(temp_dir, file.filename)
            with open(path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_paths.append(path)
            
        if not saved_paths:
            raise HTTPException(status_code=400, detail="No valid PDF files uploaded.")
            
        batch_result = load_and_split_multiple_pdfs(saved_paths)
        chunks = batch_result["chunks"]
        
        if chunks:
            add_documents_to_vectorstore(chunks)
            
        return IngestResponse(
            files_processed=batch_result["processed_files"],
            chunks_processed=len(chunks),
            message=f"Batch ingestion complete. Ingested {len(batch_result['processed_files'])} PDFs."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch ingestion failed: {str(e)}")
    finally:
        for p in saved_paths:
            if os.path.exists(p):
                os.remove(p)

@app.post("/api/v1/ingest/directory", response_model=IngestResponse)
def ingest_directory(request: DirectoryIngestRequest):
    """Ingest an entire local folder/directory containing 100+ PDF documents."""
    try:
        batch_result = load_and_split_directory(request.directory_path)
        chunks = batch_result["chunks"]
        
        if not chunks:
            raise HTTPException(status_code=404, detail="No PDF files or chunks found in specified directory.")
            
        add_documents_to_vectorstore(chunks)
        
        return IngestResponse(
            files_processed=batch_result["processed_files"],
            chunks_processed=len(chunks),
            message=f"Directory ingestion complete. Ingested {len(batch_result['processed_files'])} PDFs from folder."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Directory ingestion failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
