from unittest.mock import MagicMock

from app.generation.llm_service import LLMService


def test_generate_returns_stripped_text(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake")
    service = LLMService()

    fake = MagicMock()
    fake.choices = [MagicMock(message=MagicMock(content="  ok  "))]
    service.client = MagicMock()
    service.client.chat.completions.create.return_value = fake

    assert service.generate("question") == "ok"