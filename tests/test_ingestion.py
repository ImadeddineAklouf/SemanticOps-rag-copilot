from app.ingestion.ingestion_service import IngestionService


def test_ingest_text_file(tmp_path, store, embeddings):
    doc = tmp_path / "note.txt"
    doc.write_text("SemanticOps indexe la documentation technique. " * 20, encoding="utf-8")

    service = IngestionService(vector_store=store, embedding_service=embeddings)
    created = service.ingest(str(doc))

    assert created > 0
    assert store.collection.count() == created