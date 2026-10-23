from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import embedding_service, vector_store
from app.api.schemas import QueryRequest, QueryResponse
from app.generation.rag_pipeline import RAGPipeline
from app.monitoring.metrics import metrics
from app.retrieval.retriever import Retriever

router = APIRouter()


@lru_cache
def get_pipeline() -> RAGPipeline:
    """Build the RAG pipeline on first use (so the app starts without a key)."""
    try:
        return RAGPipeline(
            retriever=Retriever(
                vector_store=vector_store,
                embedding_service=embedding_service,
            )
        )
    except ValueError as exc:
        # Raised by LLMService when GROQ_API_KEY is missing.
        raise HTTPException(503, str(exc)) from exc


@router.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    metrics.register_query()
    try:
        answer = pipeline.ask(request.question)
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc
    return QueryResponse(answer=answer)
