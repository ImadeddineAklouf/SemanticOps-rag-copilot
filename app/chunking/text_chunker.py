from app.domain.chunk import Chunk
from app.domain.document import Document
from app.core.config import settings

class TextChunker:
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.chunk_size = (
            chunk_size
            if chunk_size is not None
            else settings.chunking.chunk_size
        )
        self.chunk_overlap = (
            chunk_overlap
            if chunk_overlap is not None
            else settings.chunking.chunk_overlap
        )

        if self.chunk_size <= 0:
            raise ValueError("chunk_size doit être strictement positif")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(
                "chunk_overlap doit être compris entre 0 et chunk_size - 1"
            )

    def chunk(
        self,
        document: Document,
    ) -> list[Chunk]:

        chunks = []

        text = document.content

        start = 0
        index = 0

        while start < len(text):

            end = start + self.chunk_size

            chunk_text = text[start:end]

            chunks.append(
                Chunk(
                    id=f"{document.id}_{index}",
                    document_id=document.id,
                    content=chunk_text,
                    chunk_index=index,
                    source=document.source,
                    file_type=document.source.split(".")[-1],
                )
            )

            start += (
                self.chunk_size
                - self.chunk_overlap
            )

            index += 1

        return chunks