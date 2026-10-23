from app.chunking.text_chunker import TextChunker
from app.embeddings.embedding_service import EmbeddingService
from app.loaders.factory import LoaderFactory
from app.monitoring.metrics import metrics
from app.vectorstore.chroma_store import ChromaStore


class IngestionService:
    def __init__(
        self,
        vector_store: ChromaStore,
        embedding_service: EmbeddingService,
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.chunker = TextChunker()

    def ingest(self, file_path: str) -> int:
        loader = LoaderFactory.get_loader(file_path)
        document = loader.load(file_path)

        chunks = self.chunker.chunk(document)
        if not chunks:
            return 0

        # Re-ingesting a document replaces its previous chunks.
        self.vector_store.delete_document(document.id)

        embeddings = self.embedding_service.embed_batch(
            [chunk.content for chunk in chunks]
        )
        self.vector_store.add_chunks(chunks, embeddings)

        metrics.register_ingestion(len(chunks))
        return len(chunks)
