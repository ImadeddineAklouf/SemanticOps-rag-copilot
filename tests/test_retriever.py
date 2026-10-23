from app.retrieval.retriever import Retriever
from tests.conftest import make_chunk


def test_retrieval_returns_closest_chunk(store, embeddings):
    text = "SemanticOps platform"
    other = "Une phrase complètement différente"
    store.add_chunk(make_chunk(text, "1"), embeddings.embed(text))
    store.add_chunk(make_chunk(other, "2"), embeddings.embed(other))

    retriever = Retriever(store, embeddings)
    results = retriever.retrieve(text, top_k=1)

    assert results["documents"][0][0] == text