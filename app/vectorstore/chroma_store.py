import chromadb

class ChromaStore:
    def __init__(
        self,
        collection_name: str = "semanticops",
        path: str = "data/vector_db",
    ):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks, embeddings):
        self.collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.content for c in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "document_id": c.document_id,
                    "source": c.source or "",
                    "file_type": c.file_type or "",
                    "chunk_index": c.chunk_index,
                }
                for c in chunks
            ],
        )

    def add_chunk(self, chunk, embedding):
        self.add_chunks([chunk], [embedding])

    def delete_document(self, document_id: str):
        self.collection.delete(where={"document_id": document_id})

    def search(self, embedding, top_k: int = 5):
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )
