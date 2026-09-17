from typing import Optional, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from config import GOOGLE_API_KEY, LLM_MODEL
from vector_store import get_retriever

DEFAULT_QA_PROMPT = """Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}"""

SUMMARIZATION_PROMPT = """You are an expert document summarizer. Synthesize and summarize the provided context clearly and thoroughly.

Context:
<context>
{context}
</context>

Task: {input}

Structure your response with:
1. **Executive Summary**: A concise overview of the core points.
2. **Key Takeaways & Details**: Bulleted list of key findings, data, or arguments.
3. **Conclusion / Recommendations**: Final summary statement.
"""

def get_llm() -> ChatGoogleGenerativeAI:
    """Initialize ChatGoogleGenerativeAI LLM."""
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2
    )

def create_rag_chain(filename_filter: Optional[str] = None, k: int = 5):
    """Build and return Q&A retrieval chain with optional filename filtering."""
    llm = get_llm()
    retriever = get_retriever(search_type="similarity", k=k, filename_filter=filename_filter)
    
    prompt = ChatPromptTemplate.from_template(DEFAULT_QA_PROMPT)
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)

def create_summarization_chain(filename_filter: Optional[str] = None, k: int = 15):
    """Build and return summarization chain with MMR retriever for broad context coverage."""
    llm = get_llm()
    # MMR (Maximal Marginal Relevance) fetches diverse chunks across documents
    retriever = get_retriever(search_type="mmr", k=k, filename_filter=filename_filter)
    
    prompt = ChatPromptTemplate.from_template(SUMMARIZATION_PROMPT)
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)

def ask_question(query: str, filename_filter: Optional[str] = None) -> Dict[str, Any]:
    """Query the RAG chain and return response dictionary."""
    chain = create_rag_chain(filename_filter=filename_filter, k=5)
    result = chain.invoke({"input": query})
    return {
        "answer": result.get("answer", ""),
        "context": [doc.page_content for doc in result.get("context", [])],
        "sources": sorted(list(set(doc.metadata.get("source", "Unknown") for doc in result.get("context", []))))
    }

def summarize_documents(
    topic_or_query: Optional[str] = None,
    filename_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate structured summary for a specific document or across all 100+ ingested PDFs.
    """
    if filename_filter:
        prompt_task = f"Provide a comprehensive summary of the document '{filename_filter}'."
    elif topic_or_query:
        prompt_task = f"Summarize all information related to '{topic_or_query}' across the ingested documents."
    else:
        prompt_task = "Provide an executive summary of the overall document corpus."
        
    chain = create_summarization_chain(filename_filter=filename_filter, k=15)
    result = chain.invoke({"input": prompt_task})
    
    return {
        "summary": result.get("answer", ""),
        "sources_used": sorted(list(set(doc.metadata.get("source", "Unknown") for doc in result.get("context", [])))),
        "chunks_analyzed": len(result.get("context", []))
    }
