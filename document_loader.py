import os
import glob
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP

def load_pdf(file_path: str) -> List[Document]:
    """Load a single PDF document and attach normalized source metadata."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # Normalize source filename metadata
    filename = os.path.basename(file_path)
    for doc in docs:
        doc.metadata["source"] = filename
        doc.metadata["file_path"] = file_path
    return docs

def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller chunks for vector storage."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.split_documents(documents)

def load_and_split_pdf(file_path: str) -> List[Document]:
    """Load single PDF and return split document chunks with metadata."""
    docs = load_pdf(file_path)
    return split_documents(docs)

def load_and_split_multiple_pdfs(file_paths: List[str]) -> Dict[str, Any]:
    """Batch process multiple PDF files and return combined chunks and file stats."""
    all_chunks: List[Document] = []
    processed_files = []
    failed_files = []
    
    for file_path in file_paths:
        try:
            chunks = load_and_split_pdf(file_path)
            all_chunks.extend(chunks)
            processed_files.append(os.path.basename(file_path))
        except Exception as e:
            failed_files.append({"file": os.path.basename(file_path), "error": str(e)})
            
    return {
        "chunks": all_chunks,
        "processed_files": processed_files,
        "failed_files": failed_files
    }

def load_and_split_directory(directory_path: str) -> Dict[str, Any]:
    """Scan a directory for all .pdf files (supports 100+ PDFs) and split into chunks."""
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    pdf_pattern = os.path.join(directory_path, "**", "*.pdf")
    pdf_files = glob.glob(pdf_pattern, recursive=True)
    
    return load_and_split_multiple_pdfs(pdf_files)
