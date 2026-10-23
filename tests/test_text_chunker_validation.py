import pytest

from app.chunking.text_chunker import TextChunker


def test_overlap_must_be_lower_than_chunk_size():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=100)


def test_zero_overlap_is_respected():
    chunker = TextChunker(chunk_size=100, chunk_overlap=0)
    assert chunker.chunk_overlap == 0


def test_chunk_size_must_be_positive():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=0, chunk_overlap=0)
