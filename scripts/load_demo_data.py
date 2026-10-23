"""Ingest the sample documents from docs/ into the vector database.

Usage (from the project root):
    python -m scripts.load_demo_data
"""
from pathlib import Path

from app.core.config import settings
from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.ingestion_service import IngestionService
from app.vectorstore.chroma_store import ChromaStore


def main() -> None:
    store = ChromaStore(collection_name=settings.vectorstore.collection_name)
    service = IngestionService(store, EmbeddingService())

    for path in sorted(Path("docs").glob("*.txt")):
        created = service.ingest(str(path))
        print(f"{path.name}: {created} chunks")

    print("Data loaded")


if __name__ == "__main__":
    main()
