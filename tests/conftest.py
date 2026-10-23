import hashlib

import pytest

from app.domain.chunk import Chunk
from app.vectorstore.chroma_store import ChromaStore


class FakeEmbeddingService:
    def embed(self, text: str) -> list[float]:
        return [float(b) for b in hashlib.sha256(text.encode()).digest()]

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


class FakeLLM:
    def __init__(self, reply: str = "réponse de test"):
        self.reply = reply
        self.last_prompt = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.reply


@pytest.fixture
def embeddings():
    return FakeEmbeddingService()


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def store(tmp_path):
    return ChromaStore(collection_name="test", path=str(tmp_path))


def make_chunk(text: str, chunk_id: str = "1") -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="doc",
        content=text,
        chunk_index=0,
        source="doc.txt",
        file_type="txt",
    )