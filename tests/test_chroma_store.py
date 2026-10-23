from tests.conftest import make_chunk


def test_store_creation(store):
    assert store.collection is not None


def test_insert_and_search(store, embeddings):
    text = "SemanticOps"
    store.add_chunk(make_chunk(text), embeddings.embed(text))

    result = store.search(embeddings.embed(text), top_k=1)

    assert result["documents"][0][0] == text


def test_add_same_chunk_twice_does_not_duplicate(store, embeddings):
    text = "SemanticOps"
    chunk = make_chunk(text)

    store.add_chunk(chunk, embeddings.embed(text))
    store.add_chunk(chunk, embeddings.embed(text))

    assert store.collection.count() == 1


def test_delete_document(store, embeddings):
    text = "SemanticOps"
    store.add_chunk(make_chunk(text), embeddings.embed(text))

    store.delete_document("doc")

    assert store.collection.count() == 0