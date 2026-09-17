from typing import List, Optional, Dict, Any
from langchain_google_genai import GoogleGenerativeAIEmbeddings
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from langchain_core.documents import Document
from config import GOOGLE_API_KEY, CHROMA_PERSIST_DIR, EMBEDDING_MODEL

def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Initialize Google Generative AI Embeddings."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )

def get_vector_store(persist_directory: str = CHROMA_PERSIST_DIR) -> Chroma:
    """Load existing Chroma vector store or return initialized vector store."""
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

def add_documents_to_vectorstore(chunks: List[Document], persist_directory: str = CHROMA_PERSIST_DIR) -> Chroma:
    """Store document chunks in Chroma database and persist."""
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vectorstore

def get_retriever(
    search_type: str = "similarity",
    k: int = 5,
    filename_filter: Optional[str] = None,
    persist_directory: str = CHROMA_PERSIST_DIR
):
    """
    Get a configured retriever from Chroma vector store.
    Supports 'similarity' or 'mmr' (Maximal Marginal Relevance for broad coverage summarization).
    Supports filtering by specific source filename.
    """
    vectorstore = get_vector_store(persist_directory)
    search_kwargs: Dict[str, Any] = {"k": k}
    
    if filename_filter:
        search_kwargs["filter"] = {"source": filename_filter}
        
    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )

def get_indexed_documents() -> List[str]:
    """Retrieve list of unique document filenames stored in vector DB."""
    try:
        vectorstore = get_vector_store()
        collection = vectorstore._collection
        results = collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])
        
        filenames = set()
        for meta in metadatas:
            if meta and "source" in meta:
                filenames.add(meta["source"])
        return sorted(list(filenames))
    except Exception:
        return []
