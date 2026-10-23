from app.generation.rag_pipeline import RAGPipeline
from app.retrieval.retriever import Retriever
from tests.conftest import make_chunk


def test_pipeline_sends_context_to_llm(store, embeddings, llm):
    text = "RAG combines retrieval and generation."
    store.add_chunk(make_chunk(text), embeddings.embed(text))

    pipeline = RAGPipeline(Retriever(store, embeddings), llm_service=llm)

    assert pipeline.ask("What is RAG?") == "réponse de test"
    assert text in llm.last_prompt


def test_pipeline_without_documents_skips_llm(store, embeddings, llm):
    pipeline = RAGPipeline(Retriever(store, embeddings), llm_service=llm)

    assert "aucune information" in pipeline.ask("Question ?")
    assert llm.last_prompt is None