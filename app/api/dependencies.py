import os

from app.core.config import settings
from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.chroma_store import ChromaStore

# Single shared instances used by every route.
embedding_service = EmbeddingService()
vector_store = ChromaStore(
    collection_name=settings.vectorstore.collection_name,
    path=os.getenv("VECTOR_DB_PATH", "data/vector_db"),
)
